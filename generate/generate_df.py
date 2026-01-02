import os
import torch
import torch
from torch import Tensor as Tsr
from typing import List
from torch import nn
from .diffusion import create_diffusion, SpacedDiffusion
import numpy as np
from torchvision import transforms
import gc
from tqdm import tqdm
from .generate import make_dit, make_extractor, decode


def make_template_stats(dit_label:int,  
               diffusion:SpacedDiffusion, 
               model, extractor,
               latent_size:int, batch_size:int,
               device, num_temps:int, 
               use_tqdm:bool=False, reproducible=False):
    if num_temps == 0:
        return torch.tensor([], device=device)
    # os.makedirs(os.path.join(args.pivot_dir, sel_class), exist_ok=True)
    if reproducible:
        zs = torch.randn((num_temps, 4, latent_size, latent_size)).to(device)
    else:
        zs = torch.randn((num_temps, 4, latent_size, latent_size), device=device)
    # partition the noise vectors into batches
    batched_zs = zs.split(batch_size)
    batched_ys = [torch.tensor([dit_label]*z.size(0), device=device) for z in batched_zs]
    batched_y_nulls = [torch.tensor([1000]*z.size(0), device=device) for z in batched_zs]
    batched_zs = [torch.cat([z,z],0) for z in batched_zs]

    indices = list(range(diffusion.num_timesteps))[::-1]

    noises_lst_lst = dict()
    for i in indices:
        nlst = []
        for z in batched_zs:
            if reproducible:
                nlst.append(torch.randn_like(z, device='cpu').to(z.device))
            else:
                nlst.append(torch.randn_like(z))
        noises_lst_lst[i] = nlst

    if use_tqdm:
        indices = tqdm(indices)

    stats_dct = dict()
    for i in indices:
        # psample
        # print('denoise step i: ', i)
        with torch.no_grad():
            new_zs = []
            # noises = []
            cur_means = []
            cur_log_vars = []
            batched_ts = [torch.tensor([i] * z.shape[0], device=device) for z in batched_zs]
            for z, y_, y_null, t in zip(batched_zs, batched_ys, batched_y_nulls, batched_ts):
                y = torch.cat([y_, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=4.0)
                out = diffusion.p_mean_variance(model.forward_with_cfg, z, t, 
                                                clip_denoised=False, model_kwargs=model_kwargs)
                # noises.append(torch.randn_like(z))
                cur_means.append(out['mean'])
                cur_log_vars.append(out['log_variance'])
                out = None
            noises = noises_lst_lst[i]
        # print(f"Class {dit_label} memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        new_noises = noises
        new_noises = [ns.detach() for ns in new_noises]

        with torch.no_grad():
            for z, t, mean, logvar, nns in zip(batched_zs, batched_ts, cur_means, cur_log_vars, new_noises):
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(z.shape) - 1)))
                )  # no noise when t == 0
                # print('mean: ', mean.shape)
                # print('logvar: ', logvar.shape)
                # print('nns:', nns.shape)
                sample = mean + nonzero_mask * torch.exp(0.5 * logvar) * nns
                new_zs.append(sample)
            batched_zs = new_zs
        
            feats = [extractor(z_.chunk(2)[0]) for z_ in batched_zs]
            feats = torch.cat(feats,0)
            fmean = feats.mean([0,2,3]).cpu()
            fstd = feats.permute(1,0,2,3).contiguous().reshape([feats.size(1),-1]).std(1, unbiased=False).cpu()
            stats_dct[i] = {'mean': fmean, 'std':fstd}
            del feats, fmean, fstd
            gc.collect()

    return stats_dct

def compute_new_noises(extractor, noises, cur_means, cur_log_vars, cur_stats_lst, num_steps, reg_strength, cfg=True, opt_type='sgd', opt_lr=0.1, cos_strength=10.0):
    device = cur_means[0].device
    if cfg:
        batch_size = noises[0].size(0)//2
    else:
        batch_size = noises[0].size(0)

    with torch.no_grad():
        if cfg:
            noise = torch.cat([ns.chunk(2)[0] for ns in noises], 0)
            cur_mean = torch.cat([m.chunk(2)[0] for m in cur_means],0)
            cur_log_var = torch.cat([v.chunk(2)[0] for v in cur_log_vars],0)
        else:
            noise = torch.cat(noises, dim=0)
            cur_mean = torch.cat(cur_means, dim=0)
            cur_log_var = torch.cat(cur_log_vars, dim=0)

        shape = noise.shape
        target_norm = np.sqrt(np.prod(shape[1:]))
        cur_stats = cur_stats_lst[0]
        # cur_x_stats = cur_stats_lst[1]
        target_mean = cur_stats['mean'][None,:,None,None].to(device)
        target_std = cur_stats['std'][None,:,None,None].to(device)
        # x_mean = cur_x_stats['mean']
        # x_mean = x_mean.view(-1).contiguous().unsqueeze(0).to(device)
        # x_std = cur_x_stats['std']
        # x_std = x_std.view(-1).contiguous().unsqueeze(0).to(device)

        # print('target mean', target_mean)
        # print('target std', target_std)
        
        var = torch.exp(0.5*cur_log_var)

    noise_tsr = noise.detach().clone().requires_grad_()
    if opt_type=='sgd':
        opt = torch.optim.SGD([noise_tsr], opt_lr, 0.9)
    elif opt_type=='adam':
        opt = torch.optim.Adam([noise_tsr], opt_lr)
    elif opt_type=='adamw':
        opt = torch.optim.AdamW([noise_tsr], opt_lr)
    mask = torch.eye(noise_tsr.size(0)).to(noise_tsr.device).sub(1).mul(-1)
    eye = torch.eye(noise_tsr.size(0)).to(noise_tsr.device)*(-100)
    
    for stp in range(num_steps):
        opt.zero_grad()
        new_tsr_ = cur_mean + var * noise_tsr
        x = new_tsr_.view(new_tsr_.size(0),-1)
        norms = noise_tsr.view(noise_tsr.size(0),-1).norm(dim=1)
        norm_loss = ((norms-target_norm)**2).sum() + torch.abs(norms-target_norm).sum()
        #这一步是用extract过的样本所计算的损失
        new_tsr_ = extractor(new_tsr_).sub(target_mean).div(target_std)
        new_tsr = new_tsr_.view(new_tsr_.size(0),-1)
            
            
        ntsr = new_tsr.div(new_tsr.norm(dim=1,keepdim=True))
        cos_mtx = ntsr.mm(ntsr.T)
        cos_mtx = cos_mtx*mask + eye
        cos_loss = cos_mtx.max(dim=1)[0].sum()
        nch = new_tsr_.shape[1]
        scale_factor = np.prod(new_tsr_.shape[2:])
        mean_loss = new_tsr_.mean([0,2,3])
        mean_loss = (mean_loss.abs().sum() + mean_loss.square().sum())*scale_factor
        std_loss = new_tsr_.permute(1,0,2,3).contiguous().reshape([nch,-1]).std(1, unbiased=False).sub(1)
        std_loss = (std_loss.abs().sum() + std_loss.square().sum())*scale_factor
        loss = 1.0*norm_loss + cos_strength*cos_loss + reg_strength*(mean_loss + std_loss)

        loss = loss
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        noise_tsr = noise_tsr.detach()
        new_noise = noise_tsr.view(*shape)
        new_noises = new_noise.split(batch_size)
        # print('new noises: ', new_noises[0][0,0,:,:])
        # print('max: ', torch.max(new_noises[0]), 'min: ', torch.min(new_noises[0]), 'norm: ', torch.norm(new_noises[0].view(new_noises[0].size(0),-1), dim=1))
        # print('mean: ', torch.mean(new_noises[0][0]), 'std: ', torch.std(new_noises[0][0]))
        # print('new noise shape', new_noises[0].shape)
        if cfg:
            null_noises = [ns.chunk(2)[1] for ns in noises]
            new_noises = [torch.cat([nns,ns],0) for nns, ns in zip(new_noises, null_noises)]
    return new_noises

from tqdm import tqdm
def generate_codes(num_pivots:int, dit_label:int,  
               stats_dct:dict, diffusion:SpacedDiffusion, 
               model, extractor, 
               latent_size:int, batch_size:int,
               device, no_opt:bool=False, reg_strength:float=0.001,
               use_tqdm:bool=False, reproducible=False):
    if num_pivots == 0:
        return torch.tensor([], device=device)
    # os.makedirs(os.path.join(args.pivot_dir, sel_class), exist_ok=True)
    if reproducible:
        zs = torch.randn((num_pivots, 4, latent_size, latent_size)).to(device)
    else:
        zs = torch.randn((num_pivots, 4, latent_size, latent_size), device=device)
    # partition the noise vectors into batches
    batched_zs = zs.split(batch_size)
    batched_ys = [torch.tensor([dit_label]*z.size(0), device=device) for z in batched_zs]
    batched_y_nulls = [torch.tensor([1000]*z.size(0), device=device) for z in batched_zs]
    batched_zs = [torch.cat([z,z],0) for z in batched_zs]

    indices = list(range(diffusion.num_timesteps))[::-1]

    noises_lst_lst = dict()
    for i in indices:
        nlst = []
        for z in batched_zs:
            if reproducible:
                nlst.append(torch.randn_like(z, device='cpu').to(z.device))
            else:
                nlst.append(torch.randn_like(z))
        noises_lst_lst[i] = nlst

    if use_tqdm:
        indices = tqdm(indices)
    for i in indices:
        # psample
        # print('denoise step i: ', i)
        with torch.no_grad():
            new_zs = []
            # noises = []
            cur_means = []
            cur_log_vars = []
            batched_ts = [torch.tensor([i] * z.shape[0], device=device) for z in batched_zs]
            for z, y_, y_null, t in zip(batched_zs, batched_ys, batched_y_nulls, batched_ts):
                y = torch.cat([y_, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=4.0)
                out = diffusion.p_mean_variance(model.forward_with_cfg, z, t, 
                                                clip_denoised=False, model_kwargs=model_kwargs)
                # noises.append(torch.randn_like(z))
                cur_means.append(out['mean'])
                cur_log_vars.append(out['log_variance'])
                out = None
            noises = noises_lst_lst[i]
        # print(f"Class {dit_label} memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        if i!=0 and not no_opt:
            new_noises = compute_new_noises(extractor, noises, cur_means, cur_log_vars, [stats_dct[i]], 200, reg_strength, True, 'sgd', 0.1, 10.0)
        else:
            new_noises = noises
        new_noises = [ns.detach() for ns in new_noises]

        with torch.no_grad():
            for z, t, mean, logvar, nns in zip(batched_zs, batched_ts, cur_means, cur_log_vars, new_noises):
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(z.shape) - 1)))
                )  # no noise when t == 0
                # print('mean: ', mean.shape)
                # print('logvar: ', logvar.shape)
                # print('nns:', nns.shape)
                sample = mean + nonzero_mask * torch.exp(0.5 * logvar) * nns
                new_zs.append(sample)
            batched_zs = new_zs

    # del zs, batched_ys, batched_y_nulls, indices, noises_lst_lst, nlst, new_zs, cur_means, cur_log_vars, batched_ts, y, model_kwargs, out, noises, new_noises, nonzero_mask, sample
    # with torch.cuda.device(device):
    #     torch.cuda.empty_cache()
        
    with torch.no_grad():
        full_zs = torch.cat([z.chunk(2)[0] for z in batched_zs],0)
    return full_zs

from diffusers.models import AutoencoderKL
def generate_images(device, dit_path:str, 
                    dit_labels:List[int], stats_dcts_lst:List[dict], extractor:nn.Module,
                    num_pivots:int, no_opt:bool=False, reg_strength=0.0001,
                    image_size:int=256, dit_bs:int=5, vae_bs:int=10,
                    use_tqdm:bool=False, 
                    seed=0, reproduce=False
                    ):
    #prepare vae & diffusion
    diffusion = create_diffusion('50')
    dit = make_dit(device, dit_path)
    assert len(dit_labels) == len(stats_dcts_lst)
    codes_lst = []
    for stats_dct, dit_label in zip(stats_dcts_lst, dit_labels):
        torch.manual_seed(seed + dit_label)
        codes_lst.append(
            generate_codes(num_pivots, dit_label, stats_dct, diffusion, dit, extractor, 256//8,
                           dit_bs, device, no_opt, reg_strength, use_tqdm, reproduce).cpu()
        )
    del dit
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=f'sd-vae-mse').to(device)
    for pa in vae.parameters():
        pa.requires_grad_(False)
    with torch.no_grad():
        images_lst = [decode(zs.to(device), vae, None, image_size, vae_bs).cpu() for zs in codes_lst]
    return images_lst

def generate_stats(device, dit_path:str, 
                dit_labels:List[int], extractor:nn.Module,
                num_pivots:int, 
                dit_bs:int=5, 
                use_tqdm:bool=False, 
                seed=0, 
                reproduce=False,
                ):
    #prepare vae & diffusion
    diffusion = create_diffusion('50')
    dit = make_dit(device, dit_path)
    stats_dcts_lst = []
    for dit_label in dit_labels:
        torch.manual_seed(seed + dit_label)
        stats_dcts_lst.append(make_template_stats(dit_label,diffusion, dit, extractor, 256//8, dit_bs, device, 
                            num_pivots, use_tqdm, reproduce))
    return stats_dcts_lst
