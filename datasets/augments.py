# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py

import torch
import random
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

__all__ = ["Compose", "Lighting", "ColorJitter"]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec, device='cpu'):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


class CutOut():
    def __init__(self, ratio, device='cpu'):
        self.ratio = ratio
        self.device = device

    def __call__(self, x):
        n, _, h, w = x.shape
        cutout_size = [int(h * self.ratio + 0.5), int(w * self.ratio + 0.5)]
        offset_x = torch.randint(h + (1 - cutout_size[0] % 2), size=[1], device=self.device)[0]
        offset_y = torch.randint(w + (1 - cutout_size[1] % 2), size=[1], device=self.device)[0]

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(n, dtype=torch.long, device=self.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=self.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=self.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=h - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=w - 1)
        mask = torch.ones(n, h, w, dtype=x.dtype, device=self.device)
        mask[grid_batch, grid_x, grid_y] = 0

        x = x * mask.unsqueeze(1)
        return x


class Normalize():
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std

import torch.nn.functional as F
class DiffAug():
    '''
    DSA增强类

    '''
    def __init__(self,
                 strategy='color_crop_cutout_flip_scale_rotate',
                 batch=False,
                 ratio_cutout=0.5,
                 single=False):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

        self.batch = batch

        self.aug = True
        if strategy == '' or strategy.lower() == 'none':
            self.aug = False
        else:
            self.strategy = []
            self.flip = False
            self.color = False
            self.cutout = False
            for aug in strategy.lower().split('_'):
                if aug == 'flip' and single == False:
                    self.flip = True
                elif aug == 'color' and single == False:
                    self.color = True
                elif aug == 'cutout' and single == False:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            'color': [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            'crop': [self.crop_fn],
            'cutout': [self.cutout_fn],
            'flip': [self.flip_fn],
            'scale': [self.scale_fn],
            'rotate': [self.rotate_fn],
            'translate': [self.translate_fn],
        }

    def __call__(self, x, single_aug=True, seed=-1):
        if not self.aug:
            return x
        else:
            if self.flip:
                self.set_seed(seed)
                x = self.flip_fn(x, self.batch)
            if self.color:
                for f in self.aug_fn['color']:
                    self.set_seed(seed)
                    x = f(x, self.batch)
            if len(self.strategy) > 0:
                if single_aug:
                    # single，从strategy里随机选择一种aug
                    idx = np.random.randint(len(self.strategy))
                    p = self.strategy[idx]
                    for f in self.aug_fn[p]:
                        self.set_seed(seed)
                        x = f(x, self.batch)
                else:
                    # multiple，所有strategy都要
                    for p in self.strategy:
                        for f in self.aug_fn[p]:
                            self.set_seed(seed)
                            x = f(x, self.batch)
            if self.cutout:
                self.set_seed(seed)
                x = self.cutout_fn(x, self.batch)

            x = x.contiguous()
            return x

    def set_seed(self, seed):
        '''手动设置随机种子'''
        if seed > 0:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

    def scale_fn(self, x, batch=True):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale

        if batch:
            sx = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = [[sx, 0, 0], [0, sy, 0]]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            sx = np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)

        grid = F.affine_grid(theta, x.shape)
        x = F.grid_sample(x, grid)
        return x

    def rotate_fn(self, x, batch=True):
        # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.ratio_rotate

        if batch:
            theta = (np.random.uniform() - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = [[np.cos(theta), np.sin(-theta), 0], [np.sin(theta), np.cos(theta), 0]]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            theta = (np.random.uniform(size=x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = [[[np.cos(theta[i]), np.sin(-theta[i]), 0],
                      [np.sin(theta[i]), np.cos(theta[i]), 0]] for i in range(x.shape[0])]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)

        grid = F.affine_grid(theta, x.shape)
        x = F.grid_sample(x, grid)
        return x

    def flip_fn(self, x, batch=True):
        prob = self.prob_flip

        if batch:
            coin = np.random.uniform()
            if coin < prob:
                return x.flip(3)
            else:
                return x
        else:
            randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            return torch.where(randf < prob, x.flip(3), x)

    def brightness_fn(self, x, batch=True):
        # mean
        ratio = self.brightness

        if batch:
            randb = np.random.uniform()
        else:
            randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5) * ratio
        return x

    def saturation_fn(self, x, batch=True):
        # channel concentration
        ratio = self.saturation

        x_mean = x.mean(dim=1, keepdim=True)
        if batch:
            rands = np.random.uniform()
        else:
            rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def contrast_fn(self, x, batch=True):
        # spatially concentrating
        ratio = self.contrast

        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        if batch:
            randc = np.random.uniform()
        else:
            randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def translate_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad

        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_y = torch.randint(-shift_y,
                                          shift_y + 1,
                                          size=[x.size(0), 1, 1],
                                          device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def crop_fn(self, x, batch=True):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad

        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        if batch:
            translation_x = np.random.randint(-shift_x, shift_x + 1)
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_x = torch.randint(-shift_x,
                                          shift_x + 1,
                                          size=[x.size(0), 1, 1],
                                          device=x.device)

            translation_y = torch.randint(-shift_y,
                                          shift_y + 1,
                                          size=[x.size(0), 1, 1],
                                          device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1, 1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def cutout_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2))
            offset_y = np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2))
        else:
            offset_x = torch.randint(0,
                                     x.size(2) + (1 - cutout_size[0] % 2),
                                     size=[x.size(0), 1, 1],
                                     device=x.device)

            offset_y = torch.randint(0,
                                     x.size(3) + (1 - cutout_size[1] % 2),
                                     size=[x.size(0), 1, 1],
                                     device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    def cutout_inv_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = np.random.randint(0, x.size(2) - cutout_size[0])
            offset_y = np.random.randint(0, x.size(3) - cutout_size[1])
        else:
            offset_x = torch.randint(0,
                                     x.size(2) - cutout_size[0],
                                     size=[x.size(0), 1, 1],
                                     device=x.device)
            offset_y = torch.randint(0,
                                     x.size(3) - cutout_size[1],
                                     size=[x.size(0), 1, 1],
                                     device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
        mask = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 1.
        x = x * mask.unsqueeze(1)
        return x