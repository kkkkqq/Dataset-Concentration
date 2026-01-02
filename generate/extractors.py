import torch
from torch import nn


def get_extractor(name:str, in_channel:int):
    if name == 's16':
        return SimpleExtractor(in_channel)
    else:
        raise NotImplementedError

class SimpleExtractor(nn.Module):

    def __init__(self, in_channel, num_groups=16):
        super().__init__()
        widths = (16, 32, 64, 128)
        self.projector = nn.Conv2d(in_channel, widths[0]*num_groups, 3, 1, 1)
        self.act0 = nn.LeakyReLU()
        bulk_lst = []
        for wd_idx in range(len(widths)-1):
            bulk_lst.append(nn.Conv2d(widths[wd_idx]*num_groups, widths[wd_idx+1]*num_groups, 4, 2, 1, groups=num_groups))
            if wd_idx != len(widths)-2:
                bulk_lst.append(nn.LeakyReLU())
        self.bulk = nn.Sequential(*bulk_lst)
    
    def forward(self, x):
        feat = self.bulk(self.act0(self.projector(x)))
        return feat
