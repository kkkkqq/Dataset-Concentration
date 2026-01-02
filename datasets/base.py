import os
import shutil
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Callable, Tuple
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR
from .sampling import ImageFolderWithIndex

class DstConfig():

    def __init__(self):
        self.nclass:int = None
        self.name:str = None
        self.sgd_lr:float = None
        self.adamw_lr:float = None
    
    def class_for_rank(self, rank:int, num_procs:int)->Tuple[Iterable[int], Iterable[str], Iterable[int]]:
        '''return dit_labels, class_tags, classifier_labels used for this rank'''
        raise NotImplementedError

    def ipc_epoch(self, ipc:int)->int:
        raise NotImplementedError
    
    def train_trans(self, size:int)->Callable:
        raise NotImplementedError
    
    def val_trans(self)->Callable:
        raise NotImplementedError
    
    def train_dataset(self, size:int, dir:str, no_trans:bool)->Iterable:
        raise NotImplementedError
    
    def val_dataset(self, size:int, dir:str)->Iterable:
        raise NotImplementedError
    
    def default_sgd(self, params:Iterable[Tensor], lr_mul:float)->SGD:
        raise NotImplementedError
    
    def default_adamw(self, params:Iterable[Tensor], lr_mul:float)->AdamW:
        raise NotImplementedError
    
    def default_multisteplr(self, optimizer:Optimizer, epochs:int)->MultiStepLR:
        raise NotImplementedError
    
    def default_coslr(self, optimizer:Optimizer, epochs:int)->LambdaLR:
        raise NotImplementedError

    def default_cosineannealinglr(self, optimizer:Optimizer, epochs:int)->CosineAnnealingLR:
        raise NotImplementedError
    
    def default_cosineannealwithwarmuplr(self, optimizer:Optimizer, epochs:int)->CosineAnnealingLR:
        raise NotImplementedError
    
    def save_images(self, img_tsr:Tensor, path:str, value_range:Tuple[float,float]=(-1.,1.), shift:int=0):
        raise NotImplementedError
    
    def class_dataset(self, class_idx:int, size:int, dir:str)->ImageFolderWithIndex:
        raise NotImplementedError
    
    def save_new_images(self, items:Iterable, path:str):
        raise NotImplementedError
    
    def merge_dataset(self, merge_from:str, merge_to:str):
        raise NotImplementedError
    
    def cutmix_config(self)->Tuple[float,float]:
        '''mix_p, beta'''
        raise NotImplementedError
    
    def num_data(self)->int:
        raise NotImplementedError

    def check_non_duplicate(self, candidate:Iterable, document:Iterable, with_duplicate:bool=False)->Iterable[int]:
        raise NotImplementedError
    
    def get_teacher(self, name:str='resnet18')->nn.Module:
        raise NotImplementedError
    
    def default_seed(self, ipc:int, da:bool):
        return None