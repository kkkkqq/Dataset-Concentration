import os
from datasets import get_config, DstConfig
import argparse
import torch
from torch import nn
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    return None

def cleanup():
    dist.destroy_process_group()

import logging
import sys
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if logging_dir:
            fh = logging.FileHandler(f'{logging_dir}/logs.txt')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

from tqdm import tqdm
import numpy as np
import time
from utils import teacher_student_logits_idcs, compute_clarity
from datasets import define_model
def main_proc(rank, world_size, args):
    setup(rank, world_size, args.port)
    num_device = torch.cuda.device_count()
    devices = list(range(num_device))
    device = devices[rank]
    config = get_config(args.config_name)
    

    #logger
    if rank==0:
        os.makedirs(args.log_dir, exist_ok=True)
        logger_ = create_logger(args.log_dir)
    else:
        logger_ = create_logger(None)
    logger = lambda s: logger_.info(s)

    dit_labels, class_tags, classifier_labels = config.class_for_rank(rank, world_size)

    save_items = []
    if args.load_odds_from is None:
        assert args.save_odds_to is not None
        logger('odds path not given, produce new odds...')
        logger('reading classwise imagefolders...')
        template_imgs_lst = [config.class_dataset(clidx, args.image_size, args.data_path) for clidx in classifier_labels]
        logger('done\n')
        logger('reading pre-trained classifier...')
        classifier = define_model(args.net_type, args.in_size, config.nclass, args.depth, args.norm_type, logger, config.name)
        if args.load_ckpt_from is None:
            ckptname = f'{args.net_type}{args.depth}_{args.norm_type}_{args.in_size}x{args.in_size}.pt'
            clfpth = os.path.join(args.cache_dir, ckptname)
        else:
            clfpth = args.load_ckpt_from
        classifier.load_state_dict(torch.load(clfpth, map_location='cpu'))
        classifier.to(device)
        classifier.eval()
        for pa in classifier.parameters():
            pa.requires_grad_(False)
        logger('done\n')

        logger(f'reading teacher {args.teacher_name}...')
        teacher = config.get_teacher(args.teacher_name)
        if args.load_teacher_from is None:
            pass
        else:
            teacher.load_state_dict(torch.load(args.load_teacher_from, map_location='cpu'))
        teacher.to(device)
        teacher.eval()
        for pa in teacher.parameters():
            pa.requires_grad_(False)
        logger('done\n')
        logger('computing odds...')
        trans = transforms.Compose([config.val_trans(), transforms.Resize(args.in_size)])
        logzip = zip(*[teacher_student_logits_idcs(tfolder, trans, teacher, classifier, device, args.odds_bs, True) for tfolder in template_imgs_lst])
        t_template_odds_lst, template_odds_lst, template_ids_lst = [list(a) for a in logzip]
        logger('done\n')
        del classifier
        del teacher
        del logzip
        logger('removed classifier and teacher')
        logger('saving odds...')
        for ci in range(len(class_tags)):
            todds=t_template_odds_lst[ci]
            odds=template_odds_lst[ci]
            svitms = template_imgs_lst[ci].get_save_items(template_ids_lst[ci])
            save_items.append(svitms)
            nm = class_tags[ci]
            svitm = (todds, odds, svitms)
            torch.save(svitm, os.path.join(args.save_odds_to, f'odds_{nm}.pt'))
            template_ids_lst[ci] = torch.arange(len(svitms))
        del todds, odds, svitms, svitm, template_imgs_lst
    else:
        logger('loading odds...')
        t_template_odds_lst, template_odds_lst, template_ids_lst = [],[],[]
        for ci in range(len(class_tags)):
            nm = class_tags[ci]
            todds, odds, svitms = torch.load(os.path.join(args.load_odds_from, f'odds_{nm}.pt'))
            t_template_odds_lst.append(todds)
            template_odds_lst.append(odds)
            save_items.append(svitms)
            template_ids_lst.append(torch.arange(len(svitms)))
        del todds, odds, svitms
    

    if args.hard_label:
        teacher_preds = [torch.ones(id_.size(0), dtype=torch.long)*clidx for id_, clidx in zip(template_ids_lst, classifier_labels)]
    else:
        teacher_preds = [lgt.argmax(dim=1) for lgt in t_template_odds_lst]
    num_tensor = torch.zeros(config.nclass, dtype=torch.long, device=device)#number of samples for each class
    for cidx, clsidx in enumerate(classifier_labels):
        num_tensor[clsidx] += template_ids_lst[cidx].size(0)
    dist.all_reduce(num_tensor, op=dist.ReduceOp.SUM)
    args.num_total = num_tensor.sum().item()
    num_tensor = num_tensor.cpu()
    num_str = ', '.join([str(i.item()) for i in num_tensor])
    logger(f'[Full Dataset] number of data per class: {num_str}')
    logger(f'[Full Dataset] total number of data: {args.num_total}')

    wrong_ids_lst = [ids[(odds.to(device).argmax(dim=1)!=tpd.to(device)).cpu()] for ids, odds, tpd in zip(template_ids_lst, template_odds_lst, teacher_preds)]
    wrong_num_tensor = torch.zeros(config.nclass, dtype=torch.long, device=device)
    for cidx, clsidx in enumerate(classifier_labels):
        wrong_num_tensor[clsidx] += wrong_ids_lst[cidx].size(0)
    dist.all_reduce(wrong_num_tensor, op=dist.ReduceOp.SUM)
    wrong_num_str = ', '.join([f'{wn}/{n}' for wn,n in zip(wrong_num_tensor,num_tensor)])
    logger(f'[Pivots] wrong sample number: \n\t{wrong_num_str}')
    total_wrong_num = wrong_num_tensor.sum().item()
    logger(f'[Pivots] total wrong samples: {total_wrong_num}/{args.num_total}')

    logger('[New] sampling new real samples...')
    if args.random_sample:
        sorted_ids_lst = [tid[torch.randperm(tid.size(0))[args.start_sampling_from:args.num_new]] for tid in template_ids_lst]
    else:
        clarity_lst = [compute_clarity(t_pred, odd, not args.no_softmax, device) for t_pred, odd in zip(teacher_preds, template_odds_lst)]
        def full_sort_ids():
            full_clarity = torch.zeros(args.num_total, dtype=torch.float, device=device)
            for cidx, clsidx in tqdm(enumerate(classifier_labels)):
                start_idx = num_tensor[:clsidx].sum().item()
                end_idx = start_idx + clarity_lst[cidx].size(0)
                full_clarity[start_idx:end_idx] += clarity_lst[cidx].to(device)
            dist.all_reduce(full_clarity)
            # print('full clarity: \n', full_clarity)
            sorted_clarity, sorted_indices = full_clarity.sort()
            sorted_indices = sorted_indices[args.start_sampling_from*config.nclass:args.start_sampling_from*config.nclass+args.num_new*config.nclass]
            # print('sorted indices: ', sorted_indices.shape)
            sorted_ids_lst = []
            for cidx, clsidx in enumerate(classifier_labels):
                start_idx = num_tensor[:clsidx].sum().item()
                # print(f'start_idx for class {clsidx}:', start_idx)
                end_idx = start_idx + clarity_lst[cidx].size(0)
                # print(f'end_idx for class {clsidx}:', end_idx)
                idcs = (sorted_indices[torch.logical_and(sorted_indices>=start_idx, sorted_indices<end_idx)]-start_idx).cpu()
                sorted_ids_lst.append(template_ids_lst[cidx][idcs])
                # print(f'indices for class {clsidx}: ', idcs)
                print(f'{len(idcs)} new smaples in total for class {clsidx}')
            return sorted_ids_lst
        def classwise_sort_ids():
            sorted_ids_lst = [tid.to(device)[clrt.to(device).sort()[1][args.start_sampling_from:args.start_sampling_from+args.num_new]].cpu() for tid, clrt in zip(template_ids_lst, clarity_lst)]
            return sorted_ids_lst
        logger('[New] sorting for new ids...')
        if args.classwise:
            sorted_ids_lst = classwise_sort_ids()
        else:
            sorted_ids_lst = full_sort_ids()
    dist.barrier()
    if args.plot:
        logger('producing visual images of frequencies...')
        def plot_histogram():
            full_clarity = torch.zeros(args.num_total, dtype=torch.float, device=device)
            for cidx, clsidx in tqdm(enumerate(classifier_labels)):
                start_idx = num_tensor[:clsidx].sum().item()
                end_idx = start_idx + clarity_lst[cidx].size(0)
                full_clarity[start_idx:end_idx] += clarity_lst[cidx].to(device)
            dist.all_reduce(full_clarity)
            if rank == 0:
                clr_max = full_clarity.max().item()
                clr_min = full_clarity.min().item()
                print('max: ', clr_max, 'min: ', clr_min)
                nbin = 100
                bars = torch.histc(full_clarity, bins=nbin).cpu()
                from matplotlib import pyplot as plt
                bins = clr_min + torch.arange(nbin)/nbin*(clr_max-clr_min)
                plt.plot(bins, bars/bars.max())
                sum_ = 0
                perc = []
                for i,b in enumerate(bars):
                    sum_ += b
                    perc.append(sum_ / sum(bars))
                perc = torch.tensor(perc)
                # plt.close()
                plt.plot(bins, perc)
                plt.xlabel('clarity value')
                plt.ylabel('percentile')
                # plot the used clarity
                plt.savefig(os.path.join(args.cache_dir, f'clperc_ckpt_{os.path.split(clfpth)[-1][:-4]}.png' if args.plot_name is None else args.plot_name))
            dist.barrier()
            return None
        plot_histogram()
        logger('done\n')
        # if args.plot:
        #     cleanup()
        #     return None 
    logger('[New] saving images...')
    if rank == 0:
        rng = tqdm(enumerate(class_tags))
    else:
        rng = enumerate(class_tags)
    for cidx, tag in rng:
        svdir = os.path.join(args.new_dir, tag)
        os.makedirs(svdir, exist_ok=True)
        # sitms = template_imgs_lst[cidx].get_save_items(sorted_ids_lst[cidx])
        sitms = [save_items[cidx][m] for m in sorted_ids_lst[cidx]]
        config.save_new_images(sitms, svdir)
    logger('done\n')
    dist.barrier()
    if args.copy_pivots:
        logger('[Pivots] copying pivot samples to new dir...')
        if rank==0:
            config.merge_dataset(args.pivot_path, args.new_dir)
    dist.barrier()
    logger('sampling complete')
    cleanup()
    return None
    

def main(args):
    world_size = torch.cuda.device_count()
    mp.spawn(main_proc, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--pivot-path", type=str, required=True)
    parser.add_argument("--new-dir", type=str, required=True)
    parser.add_argument("--load-ckpt-from", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, required=True)
    
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)

    # where to find ckpt
    parser.add_argument("--teacher-name", type=str, default='resnet18')
    parser.add_argument("--in-size", type=int, default=224)
    parser.add_argument("--net-type", type=str, default='resnet')
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--norm-type", type=str, default='instance')

    parser.add_argument("--odds-bs", type=int, default=100)
    parser.add_argument("--num-new", type=int, default=500)
    parser.add_argument("--start-sampling-from", type=int, default=0)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--classwise", action='store_true')
    parser.add_argument("--no-softmax", action='store_true')
    parser.add_argument("--random-sample", action='store_true')
    parser.add_argument("--copy-pivots", action='store_true')
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--hard-label", action='store_true')
    parser.add_argument("--plot-name", type=str, default=None)
    parser.add_argument("--load-teacher-from", type=str, default=None)
    parser.add_argument("--load-odds-from", type=str, default=None)
    parser.add_argument("--save-odds-to", type=str, default=None)
    
    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = args.new_dir
    main(args)
    
    