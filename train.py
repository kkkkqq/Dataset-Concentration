import os
from datasets import get_config, opt_maker_for_config, scheduler_maker_for_config
import argparse
import torch
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
import torch.distributed as dist
from torch.utils.data import DataLoader

PINMEMORY=True

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
    # if dist.get_rank() == 0:  # real logger
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
    # else:  # dummy logger (does nothing)
    #     logger = logging.getLogger(__name__)
    #     logger.addHandler(logging.NullHandler())
    return logger

from utils import train
from datasets import define_model
from torch.nn import DataParallel as DP
import numpy as np
import random

def main(args):
    num_device = torch.cuda.device_count()
    devices = list(range(num_device))
    # assert args.nclass % len(devices) == 0
    device = devices[0]
    config = get_config(args.config_name)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger_ = create_logger(args.log_dir)
    logger = lambda s: logger_.info(s)
    
    classifier = define_model(args.net_type, args.in_size, config.nclass, args.depth, args.norm_type, logger, config.name)
    if args.load_ckpt_from is not None:
        classifier.load_state_dict(torch.load(args.load_ckpt_from, map_location='cpu'))
    classifier.to(device)
    classifier = DP(classifier, devices)
    if args.teach:
        teacher = config.get_teacher()
        teacher.to(device)
        teacher = DP(teacher, devices)
        teacher.eval()
        for pa in teacher.parameters():
            pa.requires_grad_(False)
    else:
        teacher = None


    train_trans = config.train_trans(args.in_size)
    val_trans = config.val_trans()
    val_dataset = config.val_dataset(args.in_size, args.val_dir)
    val_loader = DataLoader(val_dataset, args.val_batch_size, shuffle=False, num_workers=args.workers, persistent_workers=args.workers>1, pin_memory=PINMEMORY)
    pivot_dataset = config.train_dataset(args.in_size, args.pivot_dir, False)
    from datasets.data import MultiEpochDataLoader
    if args.num_pivots <= 100:
            pivot_loader = MultiEpochDataLoader(
            pivot_dataset,
            args.batch_size,
            True,
            drop_last=args.drop_last,
            num_workers=args.workers,
            persistent_workers=args.workers>1,
            pin_memory=PINMEMORY,
            strict=args.strict
        )
    else:
        pivot_loader = DataLoader(
            pivot_dataset,
            args.batch_size,
            True,
            drop_last=args.drop_last,
            num_workers=args.workers,
            persistent_workers=args.workers>1,
            pin_memory=PINMEMORY
        )
    
    optgetter = opt_maker_for_config(config, args.opt)
    schegetter = scheduler_maker_for_config(config, args.sche)
    
    logger('start training...')
    bestacc1,_ = train(
        args.in_size,
        classifier, 
        pivot_loader,
        val_loader,
        config.nclass,
        optgetter,
        schegetter,
        config.ipc_epoch(args.num_pivots) if args.epochs is None else args.epochs,
        device,
        logger,
        train_trans,
        val_trans,
        args.mixup,
        config.cutmix_config() if args.mix_p is None else (args.mix_p, config.cutmix_config()[1]),
        args.batch_size,
        True,
        args.lr_mul,
        args.use_sche,
        args.print_freq,
        args.save_best,
        args.start_eval,
        teacher
    )


    classifier.eval()
    if args.save_ckpt or args.save_ckpt_to is not None:
        if args.save_ckpt_to is None:
            ckptname = f'{args.net_type}{args.depth}_{args.norm_type}_{args.in_size}x{args.in_size}.pt'
            if args.tag is not None:
                ckptname = args.tag + ckptname
            svpth = os.path.join(args.cache_dir, ckptname)
            logger(f'saving ckpt for {ckptname}')
        else:
            svpth = args.save_ckpt_to
        stdt = classifier.module.state_dict()
        torch.save(stdt, svpth)
    return bestacc1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--pivot-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--num-pivots", type=int, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--in-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    

    parser.add_argument("--net-type", type=str, default='resnet')
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--norm-type", type=str, default='instance')

    parser.add_argument("--val-batch-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--opt", type=str, default='adamw')
    parser.add_argument("--sche", type=str, choices=['steplr', 'cos', 'cosanneal', 'cosanneal_warmup'], default='steplr')
    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--lr-mul", type=float, default=1.0)
    parser.add_argument("--no-sche", action='store_true')
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--start-eval", type=float, default=0.666667)
    parser.add_argument("--save-best", action='store_true')
    parser.add_argument("--save-ckpt", action='store_true')
    parser.add_argument("--mixup", type=str, default='cut')
    parser.add_argument("--mix-p", type=float, default=None)
    parser.add_argument("--load-ckpt-from", type=str, default=None)
    parser.add_argument("--save-ckpt-to", type=str, default=None)

    parser.add_argument("--teach", action='store_true')
    parser.add_argument("--drop-last", action='store_true')
    parser.add_argument("--strict", action='store_true')
    parser.add_argument("--tag", type=str, default=None)




    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = args.cache_dir
    args.use_sche = not args.no_sche

    main(args)
    
    