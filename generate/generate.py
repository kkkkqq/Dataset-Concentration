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

SPLITSIZE = 25
REPRODUCE=[True]

class CodeReader():

    def __init__(self, smp_dir:str, class_tags:list):
        self.smp_dir = smp_dir
        self.class_tags = class_tags
        self.paths = [os.path.join(smp_dir, f'feat_{sel_class}.pt') for sel_class in self.class_tags]
        self.cur_id = 0
    
    def __getitem__(self, index:int):
        return torch.load(self.paths[index], map_location='cpu')#.div(255.0).sub(0.5).div(0.5)

    def __iter__(self):
        self.cur_id = 0
        return self
    
    def __next__(self):
        if self.cur_id < len(self.paths):
            out = self[self.cur_id]
            self.cur_id += 1
            return out
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.paths)
    

def absn2_sum(tsr:torch.Tensor):
    return tsr.abs().sum() + tsr.square().sum()

def median_odd(tsr:torch.Tensor):
    return tsr[:,tsr.size(1)//2][:,None]

def median_even(tsr:torch.Tensor):
    piv = tsr.size(1)//2
    return ((tsr[:, piv] + tsr[:,piv-1])/2)[:,None]

def element_flatten(tsr:torch.Tensor):
    return tsr.flatten(1).T

from .extractors import get_extractor
def make_extractor(device):
    extractor = get_extractor('s16', 4)
    extractor.to(device)
    extractor.eval()
    pseuin = torch.randn((2,4,32,32), device=device)
    extractor = torch.jit.trace_module(extractor, {'forward':pseuin})
    for pa in extractor.parameters():
        pa.requires_grad_(False)
    return extractor

from .models import DiT_models
from .download import find_model
def make_dit(device, dit_path:str='DiT-XL-2-256x256.pt'):
    latent_size = 256 // 8
    model = DiT_models['DiT-XL/2'](
        input_size=latent_size,
        num_classes=1000
    ).to(device)
    ckpt_path = dit_path
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    for pa in model.parameters():
        pa.requires_grad_(False)
    model.to(device)
    return model

def make_flat_feats_t(postprocess, extractor, smps_ts):
    data_nums = [tsr.size(0) for tsr in smps_ts]
    num_data = sum(data_nums)
    assert data_nums[0]>1
    dummy:Tsr = postprocess(extractor(smps_ts[0][0].unsqueeze(0))).detach().clone()
    new_shape = dummy.size()
    new_shape = [num_data, *(new_shape[1:])]
    target_tensor = torch.zeros((int(np.prod(new_shape[1:])), num_data), dtype=dummy.dtype, device=dummy.device)
    start_idx = 0
    for dn, st in zip(data_nums, smps_ts):
        end_idx = start_idx + dn
        target_tensor[:,start_idx:end_idx].add_(element_flatten(postprocess(extractor(st)).detach()))
        start_idx = end_idx
    # assert not torch.any(target_tensor==0.0) # for debugging, remove after debug
    return target_tensor, new_shape

def batched_sub_div_(target_tsr:Tsr, mean_tsr:Tsr, std_tsr:Tsr, batch_size=256):
    nfeat, nsmp = target_tsr.size()
    assert mean_tsr.size(0) == nfeat
    assert std_tsr.size(0) == nfeat
    slice_sizes = [idx.size(0) for idx in torch.arange(nfeat).split(batch_size)]
    start_idx = 0
    for ss in slice_sizes:
        end_idx = start_idx + ss
        target_tsr[start_idx:end_idx,:].sub_(mean_tsr[start_idx:end_idx,:]).div_(std_tsr[start_idx:end_idx,:])
        start_idx = end_idx
    return target_tsr

def batched_sort_(target_tsr:Tsr, batch_size=256):
    nfeat, nsmp = target_tsr.size()
    slice_sizes = [idx.size(0) for idx in torch.arange(nfeat).split(batch_size)]
    start_idx = 0
    for ss in slice_sizes:
        end_idx = start_idx + ss
        sorted = target_tsr[start_idx:end_idx,:].sort(dim=1)[0]
        target_tsr[start_idx:end_idx,:].mul_(0).add_(sorted)
        start_idx = end_idx
    return target_tsr

def compute_new_noises(noises:List[Tsr], cur_means:List[Tsr], cur_log_vars:List[Tsr], batched_ts:List[Tsr], 
                       smps_0:torch.Tensor, extractor:nn.Module, diffusion:SpacedDiffusion, 
                       sort_strength:float):
    device = cur_means[0].device
    batch_size = noises[0].size(0)//2
    postprocess = nn.AdaptiveAvgPool2d((1,1))
    postprocess.to(device)
    flatten_ = element_flatten

    with torch.no_grad():
        noise = torch.cat([ns.chunk(2)[0] for ns in noises], 0)
        cur_mean = torch.cat([m.chunk(2)[0] for m in cur_means],0)
        cur_log_var = torch.cat([v.chunk(2)[0] for v in cur_log_vars],0)

        shape = noise.shape
        target_norm = np.sqrt(np.prod(shape[1:]))
        var = torch.exp(0.5*cur_log_var)

        #compute target statistics
        smps_0 = smps_0.to(device)
        smps_ts = []
        t = batched_ts[0][0].repeat(smps_0.size(0))
        #至少5次随机加噪
        diffuse_noises = batched_latin_noise(smps_0.size(0), 5, shape[1:], device, torch.float).permute(1,0,2,3,4)
        # print('noise shape:', diffuse_noises.shape)
        # print('max val:', diffuse_noises.max())
        # print('min val:', diffuse_noises.min())
        # print('noise samples:', diffuse_noises[0][0])
        for mi in range(5):
            smps_ts.extend(diffusion.q_sample(smps_0, t, diffuse_noises[mi].detach().clone()).split(SPLITSIZE))
            # smps_ts.extend(diffusion.q_sample(smps_0, t).split(SPLITSIZE))
        del diffuse_noises
        num_mul = 5
        while sum([tsr.size(0) for tsr in smps_ts]) < shape[0]:
            smps_ts.extend(diffusion.q_sample(smps_0, t).split(SPLITSIZE))
            num_mul += 1
        # print('num_mul: ', num_mul)
        #用随机样本补全到整数倍 在10IPC和50IPC时不会触发
        if (num_mul*smps_0.size(0)) % shape[0]!=0:
            full_num = ((num_mul*smps_0.size(0)) // shape[0] + 1)*shape[0]
            num_to_fill = full_num- num_mul*smps_0.size(0) #here you assumed smps_0.size(0) must be greater than num_to_fill
            # print('full num: ', full_num)
            # print('num_to_fill: ', num_to_fill)
            if num_to_fill <= smps_0.size(0):
                perm = torch.randperm(smps_0.size(0))[:num_to_fill].to(device)
            else:
                num_left = num_to_fill % smps_0.size(0)
                num_full = num_to_fill // smps_0.size(0)
                perm = []
                for _ in range(num_full):
                    perm.append(torch.arange(smps_0.size(0), dtype=torch.long, device=device))
                perm.append(torch.randperm(smps_0.size(0))[:num_left].to(device))
                perm = torch.cat(perm,0)
            smps_ts.extend(diffusion.q_sample(smps_0[perm], t[perm]).split(SPLITSIZE))

        # feats_t = torch.cat([postprocess(extractor(smps_t)) for smps_t in smps_ts], 0)
        flat_feats_t, f_shape = make_flat_feats_t(postprocess, extractor, smps_ts)
        # f_shape = feats_t.size()
        # flat_feats_t = flatten_(feats_t)
        del smps_ts, smps_0

        normalize_mean = flat_feats_t.mean(1, keepdim=True)
        normalize_std = flat_feats_t.std(1, unbiased=False, keepdim=True)
        # flat_feats_t = flat_feats_t.sub(normalize_mean).div(normalize_std)#.cpu()
        flat_feats_t = batched_sub_div_(flat_feats_t, normalize_mean, normalize_std)
        # print('flat feats t shape: ', flat_feats_t.shape)
        # dimwise_sorted = torch.cat([ds.sort(dim=1)[0] for ds in flat_feats_t.split(1024)], 0)
        dimwise_sorted = batched_sort_(flat_feats_t)
        dimwise_pivots = torch.cat([smp.mean(1, keepdim=True) for smp in dimwise_sorted.chunk(shape[0], 1)], 1)
        del dimwise_sorted, flat_feats_t
        
    # noise = noise.mul(1.0)
    # noise = latin_noise(noise.shape[0], noise.shape[1:], device, torch.float)
    noise_tsr = noise.detach().clone().requires_grad_()
    nsmp = noise_tsr.size(0)
    opt = torch.optim.SGD([noise_tsr], 0.1, 0.9)
    
    for stp in range(200):
        opt.zero_grad()
        #norm loss
        norms = noise_tsr.view(shape[0],-1).norm(dim=1)
        norm_loss = absn2_sum(norms-target_norm)

        #compute and normalize
        new_tsr_ = cur_mean + var * noise_tsr
        x = postprocess(extractor(new_tsr_))

        #sorted losses
        x_flat = flatten_(x)
        x_flat = x_flat.sub(normalize_mean).div(normalize_std)
        x_flat_sorted = x_flat.sort(dim=1)[0]
        sort_loss = absn2_sum(x_flat_sorted - dimwise_pivots) 
        

        loss = sort_loss*sort_strength*nsmp + norm_loss
        # print('norm loss: ', norm_loss, ', reg_loss: ', reg_loss, ', repel_loss: ', repel_loss)
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
        null_noises = [ns.chunk(2)[1] for ns in noises]
        new_noises = [torch.cat([nns,ns],0) for nns, ns in zip(new_noises, null_noises)]
    return new_noises

from tqdm import tqdm
def generate_codes(num_pivots:int, dit_label:int,  
               temp_smp:Tsr, diffusion:SpacedDiffusion, 
               model, extractor,
               latent_size:int, batch_size:int,
               device, no_opt:bool=False, sort_strength:float=0.005,
               use_tqdm:bool=False, latin=False):
    if num_pivots == 0:
        return torch.tensor([], device=device)
    # os.makedirs(os.path.join(args.pivot_dir, sel_class), exist_ok=True)
    if latin:
        zs = latin_noise(num_pivots, (4, latent_size, latent_size), device, torch.float, True)
    else:
        if REPRODUCE[0]:
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
            if REPRODUCE[0]:
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
            new_noises = compute_new_noises(noises, cur_means, cur_log_vars, batched_ts, temp_smp, extractor, diffusion, sort_strength)
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


def decode(full_zs:Tsr, vae, device=None, size:int=None, batch_size:int=10):
    if device is not None:
        full_zs = full_zs.to(device)
    if size is None:
        rsz = transforms.Compose([])
    else:
        rsz =transforms.Resize(size)
    samples = []
    full_zs = full_zs.split(batch_size)
    with torch.no_grad():
        for zidx, z in enumerate(full_zs):
            samples.append(rsz(vae.decode(z / 0.18215).sample.cpu()))
        samples = torch.cat(samples, 0)
    return samples

from diffusers.models import AutoencoderKL
def generate_images(device, dit_path:str, 
                    dit_labels:List[int], template_codes_lst:List[Tsr],
                    num_pivots:int, no_opt:bool=False, sort_strength=0.005,
                    image_size:int=256, dit_bs:int=5, vae_bs:int=10,
                    use_tqdm:bool=False, 
                    latin=False, seed=0, reproduce=False
                    ):
    #prepare vae & diffusion
    REPRODUCE[0] = reproduce
    diffusion = create_diffusion('50')
    dit = make_dit(device, dit_path)
    assert len(dit_labels) == len(template_codes_lst)
    codes_lst = []
    for smps_0, dit_label in zip(template_codes_lst, dit_labels):
        torch.manual_seed(seed + dit_label)
        codes_lst.append(
            generate_codes(num_pivots, dit_label, smps_0, diffusion, dit, make_extractor(device), 256//8,
                           dit_bs, device, no_opt, sort_strength, use_tqdm, latin).cpu()
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


def lhs_normal_fast_batch(batch_size, n_samples, n_dims, device, dtype=torch.float32, eps=1e-7, lattice=False):
    """
    批处理版本的快速LHS正态采样
    
    参数:
        batch_size: 批次大小
        n_samples: 每个批次的样本数
        n_dims: 维度数
    
    返回:
        [batch_size, n_samples, n_dims] 的正态分布样本
    """
    # 生成 [batch_size, n_dims, n_samples] 的随机矩阵
    if REPRODUCE[0]:
        random_matrix = torch.rand(batch_size, n_dims, n_samples, dtype=dtype).to(device)
    else:
        random_matrix = torch.rand(batch_size, n_dims, n_samples, dtype=dtype, device=device)
    
    # 对每个批次、每个维度进行排序
    permutations = torch.argsort(random_matrix, dim=-1).to(dtype)
    
    # 添加随机偏移
    if lattice:
        offsets = 0.5
    else:
        if REPRODUCE[0]:
            offsets = torch.rand(batch_size, n_dims, n_samples, dtype=dtype).to(device)
        else:
            offsets = torch.rand(batch_size, n_dims, n_samples, dtype=dtype, device=device)
    # offsets = torch.rand(batch_size, n_dims, n_samples, dtype=dtype, device=device)
    # offsets = 1e-7 + (1-1e-7)*offsets

    # 生成[0,1]上的均匀LHS样本
    uniform_lhs = (permutations + offsets) / n_samples
    uniform_lhs = torch.clamp(uniform_lhs, eps, 1-eps)
    
    # 转换到正态分布
    # from torch.distributions import Normal
    # normal_dist = Normal(torch.tensor(0.0, device=device), 
    #                     torch.tensor(1.0, device=device))
    # normal_lhs = normal_dist.icdf(uniform_lhs)
    normal_lhs = torch.erfinv(2 * uniform_lhs - 1) * np.sqrt(2)
    
    # 转置得到 [batch_size, n_samples, n_dims]
    return normal_lhs.permute(0, 2, 1)

def lhs_normal_fast(n_samples, n_dims, device, dtype=torch.float32, eps=1e-7, lattice=False):
    """
    最快的LHS正态分布采样 (100% PyTorch)
    
    参数:
        n_samples: 样本数
        n_dims: 维度数
        device: 设备 ('cuda' 或 'cpu')
        dtype: 数据类型
        mean: 均值 (标量或向量)
        std: 标准差 (标量或向量)
    
    返回:
        [n_samples, n_dims] 的正态分布样本
    """
    # 方法1: 使用argsort技巧 (最快)
    # 生成随机矩阵 [n_dims, n_samples]
    if REPRODUCE[0]:
        random_matrix = torch.rand(n_dims, n_samples, dtype=dtype).to(device)
    else:
        random_matrix = torch.rand(n_dims, n_samples, dtype=dtype, device=device)
    
    # 对每一维排序，得到0到n_samples-1的随机排列
    permutations = torch.argsort(random_matrix, dim=1).to(dtype)
    
    # 添加随机偏移
    if lattice:
        offsets = 0.5
    else:
        if REPRODUCE[0]:
            offsets = torch.rand(n_dims, n_samples, dtype=dtype).to(device)
        else:
            offsets = torch.rand(n_dims, n_samples, dtype=dtype, device=device)
    # offsets = torch.rand(n_dims, n_samples, dtype=dtype, device=device)
    
    # 生成[0,1]上的均匀LHS样本
    uniform_lhs = (permutations + offsets) / n_samples
    uniform_lhs = torch.clamp(uniform_lhs, eps, 1-eps)
    
    # 转换到正态分布: 使用逆正态CDF
    # Φ^{-1}(u) = √2 * erfinv(2u - 1)
    # from torch.distributions import Normal
    # normal_dist = Normal(torch.tensor(0.0, device=device), 
    #                     torch.tensor(1.0, device=device))
    # normal_lhs = normal_dist.icdf(uniform_lhs)
    normal_lhs = torch.erfinv(2 * uniform_lhs - 1) * np.sqrt(2)
    
    # 转置并应用缩放
    samples = normal_lhs.t()  # [n_samples, n_dims]
    
    return samples

def batched_latin_noise(batch_size, n_samples, sample_shape, device, dtype=torch.float32, lattice=False)->Tsr:
    n_dims = np.prod(sample_shape)
    flat_samples = lhs_normal_fast_batch(batch_size, n_samples, n_dims, device, dtype, lattice=lattice)
    return flat_samples.reshape(batch_size, n_samples, *sample_shape)

def latin_noise(n_samples, sample_shape, device, dtype=torch.float32, lattice=False)->Tsr:
    n_dims = np.prod(sample_shape)
    flat_samples = lhs_normal_fast(n_samples, n_dims, device, dtype, lattice=lattice)
    return flat_samples.reshape(n_samples, *sample_shape)