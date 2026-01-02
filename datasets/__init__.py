from .sampling import ImageFolderWithIndex, TensorDatasetWithIndex
from .base import DstConfig
from .imagenet import ImageNet1kConfig, ImageNetteConfig, ImageWoofConfig
from .cifar import CIFAR10Config, CIFAR100Config
from . import data
from .train_models import define_model

def get_config(name:str)->DstConfig:
    name = name.lower()
    if name in ['imagenet1k', 'imagenet-1k', 'imagenet', 'imagenet1000']:
        return ImageNet1kConfig()
    elif name in ['imagenette', 'nette']:
        return ImageNetteConfig()
    elif name in ['imagewoof', 'woof']:
        return ImageWoofConfig()
    elif name in ['cifar10']:
        return CIFAR10Config()
    elif name in ['cifar100']:
        return CIFAR100Config()
    else:
        raise NotImplementedError
    
def opt_maker_for_config(config:DstConfig, opt_name:str='adamw'):
    opt_name = opt_name.lower()
    if opt_name == 'sgd':
        return config.default_sgd
    elif opt_name == 'adamw':
        return config.default_adamw
    else:
        raise NotImplementedError

def scheduler_maker_for_config(config:DstConfig, sche_name:str='adamw'):
    sche_name = sche_name.lower()
    if sche_name == 'cos':
        return config.default_coslr
    elif sche_name in ['steplr', 'multisteplr', 'step', 'multistep']:
        return config.default_multisteplr
    elif sche_name == 'cosanneal':
        return config.default_cosineannealinglr
    elif sche_name == 'cosanneal_warmup':
        return config.default_cosineannealwithwarmuplr
    else:
        raise NotImplementedError