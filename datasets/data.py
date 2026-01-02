import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class ImageFolder(datasets.DatasetFolder):
    """Dataset class for loading subsets with specified IPC.
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 load_memory=False,
                 load_transform=None,
                 nclass=100,
                 phase=0,
                 slct_type='random',
                 ipc=-1,
                 seed=-1,
                 spec='none',
                 return_origin=False):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        self.spec = spec
        self.return_origin = return_origin
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(nclass=nclass,
                                                                   phase=phase,
                                                                   seed=seed)
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.original_labels = self.find_original_classes()
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.original_targets = [self.original_labels[s[1]] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        """Finds the class folders in a dataset.
        """
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == 0:
            if self.spec == 'woof':
                file_list = './misc/class_woof.txt'
            elif self.spec == 'nette':
                file_list = './misc/class_nette.txt'
            elif 'woof' in self.spec and '_' in self.spec:
                file_list = './misc/class_woof.txt'
            elif 'imagenet1k' == self.spec:
                file_list = './misc/class_indices.txt'
            else:
                file_list = './misc/class100.txt'
            with open(file_list, 'r') as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split('\n')[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        if 'woof' in self.spec and '_' in self.spec:
            idx = int(self.spec.split('_')[-1])
            assert idx >=0 and idx <=9
            classes = [classes[idx]]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass

        return classes, class_to_idx

    def find_original_classes(self):
        all_classes = sorted(os.listdir(self.root))
        original_labels = []
        for class_name in self.classes:
            original_labels.append(all_classes.index(class_name))
        return original_labels

    def _subset(self, slct_type='random', ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc <= min_class

        if slct_type == 'random':
            indices = np.arange(n)
        else:
            raise AssertionError(f'selection type does not exist!')

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory
        """
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end='\r')

        print(" " * 50, end='\r')
        return imgs

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
            return sample, target, original_target
        else:
            return sample, target


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class ClassBatchSampler(object):
    """Intra-class batch sampler 
    """
    def __init__(self, cls_idx, batch_size, drop_last=True):
        self.samplers = []
        for indices in cls_idx:
            n_ex = len(indices)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                          batch_size=min(n_ex, batch_size),
                                                          drop_last=drop_last)
            self.samplers.append(iter(_RepeatSampler(batch_sampler)))

    def __iter__(self):
        while True:
            for sampler in self.samplers:
                yield next(sampler)

    def __len__(self):
        return len(self.samplers)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device('cpu'):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target


class ClassDataLoader(MultiEpochsDataLoader):
    """Basic class loader (might be slow for processing data)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(self.dataset)):
            self.cls_idx[self.dataset.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)

        self.cls_targets = torch.tensor([np.ones(self.batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device='cuda')

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.dataset[i][0] for i in indices])
        target = torch.tensor([self.dataset.targets[i] for i in indices])
        return data.cuda(), target.cuda()

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class ClassMemDataLoader():
    """Class loader with data on GPUs
    """
    def __init__(self, dataset, batch_size, drop_last=False, device='cuda'):
        self.device = device
        self.batch_size = batch_size

        self.dataset = dataset
        self.data = [d[0].to(device) for d in dataset]  # uint8 data
        self.targets = torch.tensor(dataset.targets, dtype=torch.long, device=device)

        sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(dataset))])
        self.batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                           batch_size=batch_size,
                                                           drop_last=drop_last)
        self.iterator = iter(_RepeatSampler(self.batch_sampler))

        self.nclass = dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)
        self.cls_targets = torch.tensor([np.ones(batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device=self.device)

        self.convert = None
        if self.data[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print(self.targets[indices])
        return data, self.cls_targets[c]

    def sample(self):
        indices = next(self.iterator)
        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)
        target = self.targets[indices]

        return data, target

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            data, target = self.sample()
            yield data, target


class ClassPartMemDataLoader(MultiEpochsDataLoader):
    """Class loader for ImageNet-100 with multi-processing.
       This loader loads target subclass samples on GPUs
       while can loading full training data from storage. 
    """
    def __init__(self, subclass_list, real_to_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.mem_cls = subclass_list
        self.real_to_idx = real_to_idx

        self.cls_idx = [[] for _ in range(self.nclass)]
        idx = 0
        self.data_mem = []
        print("Load target class data on memory..")
        for i in range(len(self.dataset)):
            c = self.dataset.targets[i]
            if c in self.mem_cls:
                self.data_mem.append(self.dataset[i][0].cuda())
                self.cls_idx[c].append(idx)
                idx += 1

        if self.data_mem[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)
        print(f"Subclass: {subclass_list}, {len(self.data_mem)}")

        class_batch_size = 64
        self.class_sampler = ClassBatchSampler([self.cls_idx[c] for c in subclass_list],
                                               class_batch_size,
                                               drop_last=True)
        self.cls_targets = torch.tensor([np.ones(class_batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device='cuda')

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            idx = self.real_to_idx[c]
            indices = next(self.class_sampler.samplers[idx])

        data = torch.stack([self.data_mem[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print([self.dataset.targets[i] for i in self.slct[indices]])
        return data, self.cls_targets[c]

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()

class MultiEpochDataLoader(MultiEpochsDataLoader):

    def __init__(self, *args, **kwargs):
        args, kwargs = rectify_args_kwargs(args, kwargs)
        super().__init__(*args, **kwargs)
        return None


def img_denormlaize(img, dataname='imagenet'):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = MEANS[dataname]
    std = STDS[dataname]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean


def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataname=dataname)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)


def rectify_args_kwargs(args:list, kwargs:dict):
    if len(args)<2:
        return args, kwargs
    if 'strict' in kwargs:
        if kwargs['strict']:
            kwargs.pop('strict')
            return args, kwargs
        else:
            kwargs.pop('strict')
    dividend = len(args[0])
    divider = args[1]
    if dividend % divider == 0:
        return args, kwargs
    else:
        while dividend % divider != 0 and divider > 32:
            divider -= 1
        return (args[0], divider, *args[2:]), kwargs