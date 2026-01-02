import os
import torch
from torch import nn
from torchvision import transforms
from kornia.augmentation import RandomHorizontalFlip
from .data import MEANS, STDS, ImageFolder
from torchvision.utils import save_image
from torchvision import models as thmodels
import numpy as np
from .base import DstConfig
from .sampling import ImageFolderWithIndex
from .augments import ColorJitter, Lighting

class ImageNetConfig(DstConfig):

    def __init__(self):
        super().__init__()
        self.nclass:int = None
        self.rrc_scale:tuple = (0.5, 1.0)
        self.name:str = 'imagenet'
        self.spec:str = None
        self.sgd_lr = 0.01
        self.adamw_lr = 1e-3
        
    def class_for_rank(self, rank, num_procs):
        with open('./misc/class_indices.txt', 'r') as fp:
            all_classes = fp.readlines()
        all_classes = [class_index.strip() for class_index in all_classes]
        if self.spec == 'woof':
            file_list = './misc/class_woof.txt'
        elif self.spec == 'nette':
            file_list = './misc/class_nette.txt'
        elif self.spec == 'imagenet1k':
            file_list = './misc/class_indices.txt'
        elif self.spec == 'imagenet100':
            file_list = './misc/class100.txt'
        else:
            raise NotImplementedError
        with open(file_list, 'r') as fp:
            sel_classes = fp.readlines()

        phase = rank
        class_num = int(np.ceil(self.nclass / num_procs))
        cls_from = class_num * phase
        cls_to = class_num * (phase + 1)
        cls_to = min(cls_to, self.nclass)
        sel_classes = sel_classes[cls_from:cls_to]
        sel_classes = [sel_class.strip() for sel_class in sel_classes]
        class_labels = []

        for sel_class in sel_classes:
            class_labels.append(all_classes.index(sel_class))
        dit_labels = class_labels
        class_tags = sel_classes
        classifier_labels = list(range(cls_from,cls_to))

        return dit_labels, class_tags, classifier_labels
    
    def train_dataset(self, size:int, dir:str, no_trans=False):
        nclass = self.nclass
        if size == 224:
            img_size = 256
        else:
            img_size = size
        resize_train = [transforms.Resize(img_size, antialias=True)]
        # resize_train.append(transforms.CenterCrop(img_size))
        if not no_trans:
            resize_train.append(transforms.RandomResizedCrop((size, size), self.rrc_scale, antialias=True))
        else:
            resize_train.append(transforms.CenterCrop(img_size))
        cast = [transforms.ToTensor()]
        train_trans = transforms.Compose(resize_train + cast)
        return ImageFolder(dir, train_trans, nclass=nclass, seed=0, slct_type='random', ipc=-1, load_memory=False, spec=self.spec)

    def val_dataset(self, size, dir):
        nclass = self.nclass
        if size == 224:
            img_size = 256
        else:
            img_size = size
        resize_test = [transforms.Resize(img_size, antialias=True), transforms.CenterCrop(size)]
        cast = [transforms.ToTensor()]
        test_trans = transforms.Compose(resize_test + cast)
        return ImageFolder(
                dir,
                test_trans,
                nclass=nclass,
                seed=0,
                load_memory=False,
                spec=self.spec)

    def default_sgd(self, params, lr_mul):
        return torch.optim.SGD(params,
                    self.sgd_lr*lr_mul,# 0.1 for teach, 0.01 for train
                    momentum=0.9,
                    weight_decay=1e-4)
    
    def default_adamw(self, params, lr_mul):
        return torch.optim.AdamW(params, lr=self.adamw_lr*lr_mul, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    
    def default_multisteplr(self, optimizer, epochs:int):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2 * epochs // 3, 5*epochs//6], gamma=0.2)
    
    def default_cosineannealinglr(self, optimizer, epochs):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

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

    def save_images(self, img_tsr:torch.Tensor, path:str, value_range:tuple=(-1.,1.), shift:int=0):
        os.makedirs(path, exist_ok=True)
        if len(img_tsr.shape)==3:
            img_tsr = img_tsr.unsqueeze(0)
        elif len(img_tsr.shape)==4:
            pass
        else:
            raise AssertionError(f"img tensor should be 3 or 4 dim tensor, but got {len(img_tsr.shape)}")
        for zidx, img in enumerate(img_tsr):
            save_image(img, os.path.join(path,
                    f"{zidx+shift}.png"), normalize=True, value_range=value_range)
        return None
    
    def class_dataset(self, class_idx, size, dir):
        if size == 224:
            img_size = 256
        else:
            img_size = size
        resize_train = [transforms.Resize(img_size, antialias=True)]
        resize_train.append(transforms.CenterCrop(img_size))
        cast = [transforms.ToTensor()]
        train_trans = transforms.Compose(resize_train + cast)
        return ImageFolderWithIndex(dir, train_trans, nclass=1, phase=class_idx, seed=0, slct_type='random', ipc=-1, load_memory=False, spec=self.spec)
    
    def save_new_images(self, items, path):
        os.makedirs(path, exist_ok=True)
        paths = items
        for pth in paths:
            try:
                os.link(pth, os.path.join(path, 'new'+os.path.split(pth)[-1]))
            except FileExistsError:
                print('file already exists, proceed...')
            except Exception as e:
                print('ignoring error: ')
                print(e)
            finally:
                pass
        return 
    
    def cutmix_config(self):
        return 1.0, 1.0
    
    def merge_dataset(self, merge_from, merge_to):
        _, class_tags,_ = self.class_for_rank(0,1)
        for tg in class_tags:
            mf = os.path.join(merge_from, tg)
            mt = os.path.join(merge_to, tg)
            all_files = os.listdir(mf)
            for f in all_files:
                frpth = os.path.join(mf, f)
                topth = os.path.join(mt, f)
                try:
                    os.link(frpth, topth)
                except FileExistsError as e:
                    print(e)
        return None
    
    def check_non_duplicate(self, candidate, document, with_duplicate:bool=False):
        nondupidcs = []
        dupidcs = []
        for i, itm in enumerate(candidate):
            if itm in document:
                dupidcs.append(i)
            else:
                nondupidcs.append(i)
        if with_duplicate:
            return nondupidcs, dupidcs
        else:
            return nondupidcs
    
    def get_teacher(self, name = 'resnet18'):
        teacher = thmodels.__dict__[name](pretrained=True)
        # Labels to condition the model
        with open('./misc/class_indices.txt', 'r') as fp:
            all_classes = fp.readlines()
        all_classes = [class_index.strip() for class_index in all_classes]
        if self.spec == 'woof':
            file_list = './misc/class_woof.txt'
        elif self.spec == 'nette':
            file_list = './misc/class_nette.txt'
        elif self.spec == 'imagenet1k':
            file_list = './misc/class_indices.txt'
        elif self.spec == 'imagenet100':
            file_list = './misc/class100.txt'
        else:
            raise NotImplementedError
        with open(file_list, 'r') as fp:
            sel_classes = fp.readlines()

        sel_classes = [sel_class.strip() for sel_class in sel_classes]
        class_labels = []
        
        for sel_class in sel_classes:
            class_labels.append(all_classes.index(sel_class))
        class_labels = torch.tensor(class_labels)

        try:
            model_named_parameters = [name for name, x in teacher.named_parameters()]
            for name, x in teacher.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[class_labels]
        except:
            print("ERROR in changing the number of classes, ORIGINAL CLASS LABELS USED!")
        
        for pa in teacher.parameters():
            pa.requires_grad_(False)
        
        return teacher
    
class ImageNet1kConfig(ImageNetConfig):

    def __init__(self):
        super().__init__()
        self.nclass = 1000
        self.rrc_scale = (0.5, 1.0)
        self.spec = 'imagenet1k'
        self.sgd_lr = 0.1
        self.adamw_lr = 1e-3

    def ipc_epoch(self, ipc):
        if ipc <= 100:
            return 300
        else:
            return 100
    
    def train_trans(self, size):
        aug = []
        augment = aug
        augment.append(RandomHorizontalFlip())
        augment.append(transforms.Resize(size))
        augment.append(torch.jit.script(transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet'])))
        return nn.Sequential(*augment)
    
    def val_trans(self):
        return torch.jit.script(transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet']))
    
    def num_data(self):
        return 1281167
    
class ImageNetteConfig(ImageNetConfig):

    def __init__(self):
        super().__init__()
        self.nclass = 10
        self.rrc_scale = (0.5, 1.0)
        self.spec = 'nette'
        self.sgd_lr = 0.01
        self.adamw_lr = 1e-3

    def ipc_epoch(self, ipc):
        if ipc == 1:
            epoch = 3000
        elif ipc <= 10:
            epoch = 2000
        elif ipc <= 50:
            epoch = 1500
        elif ipc <= 200:
            epoch = 1000
        else:
            epoch = 500
        return epoch
    
    def train_trans(self, size):
        jittering = ColorJitter(0.4, 0.4, 0.4)
        lightning = Lighting(
                            alphastd=0.1,
                            eigval=[0.2175, 0.0188, 0.0045],
                            eigvec=[
                                [-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203],
                            ],
                            device='cuda')
        aug = []
        aug.append(transforms.Resize(size))
        aug = aug + [RandomHorizontalFlip(), jittering, lightning]

        normal_fn = [torch.jit.script(transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet']))]
        return transforms.Compose(aug+normal_fn)
    
    def val_trans(self):
        return torch.jit.script(transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet']))
    
    def num_data(self):
        return 12894

    def default_seed(self, ipc, da):
        if da:
            if ipc == 10:
                return 4
            elif ipc == 50:
                return 4
            elif ipc == 100:
                return 3
            else:
                return None
        else:
            return None
    
class ImageWoofConfig(ImageNetteConfig):

    def __init__(self):
        super().__init__()
        self.spec = 'woof'

    def num_data(self):
        return 12454
    
    def default_seed(self, ipc, da):
        if da:
            if ipc == 10:
                return 4
            elif ipc == 50:
                return 4
            elif ipc == 100:
                return 3
            else:
                return None
        else:
            return None

