import os
from datasets import get_config, DstConfig
import argparse
import torch
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

from generate import generate_stats_df, make_extractor
def main_proc(rank, world_size, args):
    setup(rank, world_size, args.port)
    num_device = torch.cuda.device_count()
    devices = list(range(num_device))
    # assert args.nclass % len(devices) == 0
    device = devices[rank]
    config = get_config(args.config_name)
    seed = args.seed

    #logger
    if rank==0:
        os.makedirs(args.log_dir, exist_ok=True)
        logger_ = create_logger(args.log_dir)
    else:
        logger_ = create_logger(None)
    logger = lambda s: logger_.info(s)

    dit_labels, class_tags, classifier_labels = config.class_for_rank(rank, world_size)
    logger('reading template_stats...')
    extractor = make_extractor(device)
    if os.path.exists(os.path.join(args.stats_path, 'extractor.pt')):
        extractor.load_state_dict(torch.load(os.path.join(args.stats_path, 'extractor.pt')))
    else:
        os.makedirs(args.stats_path, exist_ok=True)
        torch.save(extractor.state_dict(), os.path.join(args.stats_path, 'extractor.pt'))
    stats_pths_lst = [os.path.join(args.stats_path, f'stats_{clab}.pt') for clab in classifier_labels]
    logger('done\n\n')

    logger(f'generating {args.num_pivots} template samples...')
    stats_dcts_lst =generate_stats_df(device, args.dit_path, dit_labels, extractor,
                                      args.num_pivots, args.dit_bs, args.tqdm, args.seed, not args.random)
    
    logger(f'done, saving stats to {args.stats_path}...')
    for dct, pth in zip(stats_dcts_lst, stats_pths_lst):
        torch.save(dct, pth)
    
    logger(f'generation complete')
    cleanup()
    return None

def main(args):
    world_size = torch.cuda.device_count()
    mp.spawn(main_proc, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--dit-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-pivots", type=int, default=200)
    parser.add_argument("--dit-bs", type=int, default=5)
    parser.add_argument("--tqdm", action='store_true')
    parser.add_argument("--random", action='store_true')
    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = args.stats_path
    main(args)
    
    