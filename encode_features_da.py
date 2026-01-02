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
from diffusers import AutoencoderKL

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    return None

def cleanup():
    dist.destroy_process_group()

from tqdm import tqdm
def main_proc(rank, world_size, args):
    setup(rank, world_size, args.port)
    num_device = torch.cuda.device_count()
    devices = list(range(num_device))
    # assert args.nclass % len(devices) == 0
    device = devices[rank]
    config = get_config(args.config_name)
    dit_labels, class_tags, classifier_labels = config.class_for_rank(rank, world_size)
    os.makedirs(args.feat_path, exist_ok=True)
    trans = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=f'sd-vae-mse').to(device)
    vae.eval()
    with torch.no_grad():
        for tag, lab in zip(class_tags, classifier_labels):
            fsvpth = os.path.join(args.feat_path, f'feat_{tag}.pt')
            psvpth = os.path.join(args.feat_path, f'items_{tag}.pt')
            if os.path.exists(fsvpth) and os.path.exists(psvpth):
                print('feature exists at', fsvpth, ', ignore class', tag)
                continue
            dst_class = config.class_dataset(lab, args.image_size, args.data_path)
            loader = DataLoader(dst_class, 32)
            if rank == 0:
                loader = tqdm(loader) if args.tqdm else loader
            feats = []
            idcs = []
            for tup in loader:
                img, tgt, idx = tup[0], tup[1], tup[-1]
                feats.append(vae.encode(trans(img.to(device))).latent_dist.sample().mul_(0.18215))
                idcs.append(idx.to(device))
            feats = torch.cat(feats, 0)
            idcs = torch.cat(idcs,0)
            shp = feats.shape
            tmpl = feats[:,shp[1]//2,shp[2]//2,shp[3]//2].detach().clone()
            _, idcs_idx = tmpl.sort(0)
            idcs = idcs[idcs_idx]
            feats = feats[idcs_idx]
            torch.save(feats.cpu(), fsvpth)
            if args.save_items:
                torch.save(dst_class.get_save_items(idcs.cpu()),psvpth)
            print('saved encoded features to ', fsvpth)
            del feats, idcs, idcs_idx, dst_class, loader, tmpl
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
    parser.add_argument("--feat-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--save-items", action='store_true')
    parser.add_argument("--tqdm", action='store_true')
    args = parser.parse_args()
    main(args)