import torchvision.models as thmodels
from . import resnet as RN
from . import resnet_ap as RNAP
from . import convnet as CN
from . import densenet_cifar as DN
from efficientnet_pytorch import EfficientNet

def define_model(net_type:str, in_size:int, nclass:int, depth:int, norm_type:str='instance', logger=None, dataset='imagenet'):
    """Define neural network models
    """
    net_type = net_type.lower()
    norm_type = norm_type.lower()
        
    if net_type == 'resnet':
        if norm_type=='batch' and 'imagenet' in dataset:
            if logger is not None:
                logger('detected imagenet setting, use standard ResNet from torchvision')
            return thmodels.__dict__[net_type+str(depth)](pretrained=False)
        model = RN.ResNet(dataset,
                          depth,
                          nclass,
                          norm_type=norm_type,
                          size=in_size,
                          nch=3)
    elif net_type == 'resnet_ap':
        model = RNAP.ResNetAP(dataset,
                              depth,
                              nclass,
                              width=1.0,
                              norm_type=norm_type,
                              size=in_size,
                              nch=3)
    elif net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif net_type == 'convnet':
        width = 128
        model = CN.ConvNet(nclass,
                           net_norm=norm_type,
                           net_depth=depth,
                           net_width=width,
                           channel=3,
                           im_size=(in_size, in_size))
    else:
        raise Exception('unknown network architecture: {}'.format(net_type))

    if logger is not None:
        logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")

    return model