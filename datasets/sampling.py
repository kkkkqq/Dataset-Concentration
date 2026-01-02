
from .data import ImageFolder, TensorDataset
import torchvision.datasets as datasets
import torch
from typing import Iterable
from torch import nn
from torch.utils.data import DataLoader

class ImageFolderWithIndex(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader, is_valid_file=None, load_memory=False, load_transform=None, nclass=100, phase=0, slct_type='random', ipc=-1, seed=-1, spec='none', return_origin=False):
        super().__init__(root, transform, target_transform, loader, is_valid_file, load_memory, load_transform, nclass, phase, slct_type, ipc, seed, spec, return_origin)
        self.load_memory = False

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        original_target = self.original_targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            original_target = self.target_transform(original_target)

        # Return original labels for DiT generation
        if self.return_origin:
            return sample, target, original_target, index
        else:
            return sample, target, index
    
    def get_save_items(self, indices):
        itms = []
        for i in indices:
            itms.append(self.samples[i][0])
        return itms
    
    def all_save_items(self):
        return [itm[0] for itm in self.samples]

class TensorDatasetWithIndex(TensorDataset):

    def __init__(self, images, labels, transform=None):
        super().__init__(images, labels, transform)

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)
        target = self.targets[index]
        return sample, target, index
    
    def get_save_items(self, indices):
        if isinstance(indices, torch.Tensor):
            itms = self.images[indices]
        else:
            indices = torch.tensor(indices, dtype=torch.long)
            itms = self.images[indices]
        return itms
    
    def all_save_items(self):
        return self.images

def img_tgt_idx(tup:tuple):
    return tup[0], tup[1], tup[-1]








    


    
