import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Callable, Tuple
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from tqdm import tqdm
from .sampling import TensorDatasetWithIndex
from .augments import DiffAug
from .base import DstConfig

MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MAPPING_DICT = dict(
    cifar10 = [895, 817, 13, 285, 353, 153, 31, 339, 628, 675],
    cifar100 = [948,0,431,294,337,750,309,301,671,737,809,842,839,874,321,354,653,483,329,345,423,367,892,961,314,831,119,49,504,42,3,385,110,975,277,601,333,449,104,508,846,621,288,291,44,122,834,937,665,970,673,947,708,950,986,360,977,940,717,879,460,923,989,334,357,332,387,6,646,442,946,978,356,2,338,361,682,113,53,72,335,829,985,945,532,847,528,851,292,866,466,391,936,35,894,147,903,271,489,52],
    )
CIFAR_DSTS = {'cifar10':CIFAR10, 'cifar100':CIFAR100}

class CIFARConfig(DstConfig):

    def __init__(self):
        super().__init__()
        self.name = 'cifar'
        self.sgd_lr = 0.01
        self.adamw_lr = 0.001
        self.teacher_dir = None
    
    def class_for_rank(self, rank, num_procs):
        sel_classes = [str(i) for i in range(self.nclass)]
        class_labels = list(range(self.nclass))
        class_num = int(np.ceil(self.nclass / num_procs))
        cls_from = class_num*rank
        cls_to = min(class_num*(rank+1), self.nclass)
        dit_labels = [x for x in MAPPING_DICT[self.name]]
        dit_labels = dit_labels[cls_from:cls_to]
        class_tags = sel_classes[cls_from:cls_to]
        classifier_labels = class_labels[cls_from:cls_to]
        return dit_labels, class_tags, classifier_labels
    
    def ipc_epoch(self, ipc):
        """Calculating training epochs for ImageNet
        """
        if ipc == 1:
            epoch = 3000
        elif ipc <= 10:
            epoch = 2000
        elif ipc <= 50:
            epoch = 1500
        elif ipc <= 200:
            epoch = 1000
        elif ipc <= 500:
            epoch = 500
        else:
            epoch = 300

        if self.nclass == 100:
            epoch = int((2 / 3) * epoch)
            epoch = epoch - (epoch % 100)

        return epoch
    
    def train_trans(self, size):
        aug = DiffAug('color_crop_cutout_flip_scale_rotate', batch=False)
        rsz = transforms.Resize(size, antialias=True)
        normal_fn = torch.jit.script(transforms.Normalize(mean=MEANS[self.name], std=STDS[self.name]))
        return transforms.Compose([aug, rsz, normal_fn])
    
    def val_trans(self):
        return torch.jit.script(transforms.Normalize(mean=MEANS[self.name], std=STDS[self.name]))
    
    def train_dataset(self, size, dir, no_trans):
        rsz = transforms.Resize(size)
        trans = transforms.Compose([rsz, transforms.ToTensor()])
        cvt = transforms.ConvertImageDtype(torch.float)
        try:
            dataset = CIFAR_DSTS[self.name](dir, True, trans, download=False)
        except Exception as e:
            print(f'cannot find stantard {self.name} dataset, Exception below: ')
            print(e)
            print('attempting to load classwise saved datasets...')
            imgs_lst = [rsz(cvt(torch.load(os.path.join(dir, str(c), 'tsr_dst.pt')))) for c in range(self.nclass)]
            labs_lst = [torch.zeros(im.size(0), dtype=torch.long, device='cpu') + c for im, c in zip(imgs_lst, range(self.nclass))]
            dataset = TensorDataset(torch.cat(imgs_lst,0), torch.cat(labs_lst,0))
            print('done')
        finally:
            pass
        return dataset
    
    def val_dataset(self, size, dir):
        rsz = transforms.Resize(size)
        trans = transforms.Compose([rsz, transforms.ToTensor()])
        val_dir = CIFAR_DSTS[self.name](dir, False, trans, download=False)
        return val_dir
    
    def default_sgd(self, params, lr_mul):
        return SGD(params,
                    self.sgd_lr*lr_mul,# 0.1 for teach, 0.01 for train
                    momentum=0.9,
                    weight_decay=5e-4)
    
    def default_adamw(self, params, lr_mul):
        return AdamW(params, lr=self.adamw_lr*lr_mul, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    
    def default_multisteplr(self, optimizer, epochs:int):
        return MultiStepLR(optimizer, milestones=[2*epochs// 3, 5*epochs// 6], gamma=0.1)

    def default_cosineannealinglr(self, optimizer, epochs):
        return CosineAnnealingLR(optimizer, epochs)
    
    def default_coslr(self, optimizer, epochs):
        opt = optimizer
        epch = epochs
        return torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: 0.5 * (1.0 + np.cos(np.pi * step / epch / 2))
            if step <= epch
            else 0,
            last_epoch=-1,
            )
    
    def default_cosineannealwithwarmuplr(self, optimizer, epochs):
        opt = optimizer
        epch = epochs
        return torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: 0.5 * (1.0 + np.cos(np.pi * (step-epch//10) / (epch//10*9) ))
            if step >= epch//10
            else (step+1)/epch*10,
            last_epoch=-1,
        )
    
    def save_images(self, img_tsr, path, value_range=(-1, 1), shift = 0):
        os.makedirs(path, exist_ok=True)
        if len(img_tsr.shape)==3:
            img_tsr = img_tsr.unsqueeze(0)
        elif len(img_tsr.shape)==4:
            pass
        else:
            raise AssertionError(f"img tensor should be 4 dim tensor, but got {len(img_tsr.shape)}")
        mean = sum(value_range)/2
        scale = value_range[1]-value_range[0]
        img_tsr = (img_tsr - mean)/scale
        img_tsr = img_tsr.add(0.5).mul(225.0).round().to(torch.uint8)
        torch.save(img_tsr, os.path.join(path, 'tsr_dst.pt'))
        return None
    
    def save_new_images(self, items, path):
        if not isinstance(items, Tensor):
            img_tsr = torch.cat(items, 0)
        else:
            img_tsr = items
        self.save_images(img_tsr, path, (0,1))
        return None
    
    def class_dataset(self, class_idx, size, dir):
        _, class_tags, _ = self.class_for_rank(0,1)
        path = os.path.join(dir, class_tags[class_idx])
        try:
            images = torch.load(os.path.join(path, 'tsr_dst.pt'))
        except Exception as e:
            print(f'cannot find tsr_dst.pt at {path} with following error: ')
            print(e)
            print(f'attempting to reconstruct classwise datasets at {dir}...')
            flag = os.path.join(dir, 'sampling_in_progress.pt')
            if os.path.exists(flag):
                while os.path.exists(flag):
                    time.sleep(0.5)
            else:
                torch.save(torch.tensor([]), flag)
                dst = self.train_dataset(size, dir, True)
                loader = DataLoader(dst, 500, False)
                imgs, tgts = zip(*[(img,tg) for img,tg in tqdm(loader)])
                imgs = torch.cat(imgs)
                tgts = torch.cat(tgts)
                for c in tqdm(range(self.nclass)):
                    self.save_new_images(imgs[tgts==c], os.path.join(dir, class_tags[c]))
                os.remove(flag)
            images = torch.load(os.path.join(path, 'tsr_dst.pt'))
        finally:
            pass
        
        rsz = transforms.Resize(size)
        cvt = transforms.ConvertImageDtype(torch.float)
        images = rsz(cvt(images))
        labels = torch.zeros(images.size(0), dtype=torch.long)
        return TensorDatasetWithIndex(images, labels)
    
    def cutmix_config(self):
        return 0.5, 1.0
    
    def merge_dataset(self, merge_from, merge_to):
        for c in range(self.nclass):
            frompth = os.path.join(merge_from, str(c), 'tsr_dst.pt')
            topth = os.path.join(merge_to, str(c), 'tsr_dst.pt')
            fromdst, todst = [torch.load(pth) for pth in (frompth, topth)]
            newdst = torch.cat([fromdst, todst], 0)
            torch.save(newdst, topth)
            # shutil.rmtree(os.path.join(merge_from, str(c)))

    def check_non_duplicate(self, candidate:Tensor, document:Tensor, with_duplicate:bool=False):
        candidate = candidate[:,None,:,:,:]
        document = document[None,:,:,:,:]
        ck = torch.cat([can.sub(document).abs().flatten(2).sum(dim=2) for can in candidate.split(100)],0)
        ck = ck==0.0
        ck = ck.any(dim=1)
        if with_duplicate:
            return torch.arange(ck.size(0), device=ck.device)[~ck], torch.arange(ck.size(0),device=ck.device)[ck]
        else:
            return torch.arange(ck.size(0),device=ck.device)[~ck]

    def get_teacher(self, name = 'resnet18'):
        name = name.lower()
        if 'resnet_ap' in name:
            assert name == 'resnet_ap10'
            model = define_model('resnet_ap', 32, self.nclass, 10, 'instance', None, self.name)
        elif 'resnet' in name:
            assert name == 'resnet18'
            model = define_model('resnet', 32, self.nclass, 18, 'batch', None, self.name)
        elif 'convnet' in name:
            assert name == 'convnet3'
            model = define_model('convnet', 32, self.nclass, 3, 'instance', None, self.name)
        else:
            raise NotImplementedError
        stdt = torch.load(os.path.join(self.teacher_dir, f'teacher_{name}.pt'), map_location='cpu')
        model.load_state_dict(stdt)
        for pa in model.parameters():
            pa.requires_grad_(False)
        model.eval()
        return model

        

from .train_models import define_model

class CIFAR10Config(CIFARConfig):

    def __init__(self):
        super().__init__()
        self.nclass = 10
        self.name = 'cifar10'
        self.teacher_dir = './teachers/cifar10'
    
    def num_data(self):
        return 50000
    
class CIFAR100Config(CIFARConfig):

    def __init__(self):
        super().__init__()
        self.nclass = 100
        self.name = 'cifar100'
        self.teacher_dir = './teachers/cifar100'
    
    def num_data(self):
        return 50000