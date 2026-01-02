
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as thmodels

def img_tgt_idx(tup:tuple):
    return tup[0], tup[1], tup[-1]

def logits_and_idcs(class_dataset_withidx, trans, model:nn.Module, device, bs:int=200, to_cpu:bool=False):
    loader = DataLoader(class_dataset_withidx, bs, False, num_workers=2, pin_memory=True)
    logits = []
    idcs = []
    with torch.no_grad():
        for tup in loader:
            img, tgt, idx = (tsr.to(device) for tsr in img_tgt_idx(tup))
            logits.append(model(trans(img)))
            idcs.append(idx.to(torch.long))
        logits = torch.cat(logits, 0)
        idcs = torch.cat(idcs, 0)
    idcs_sorted, odidcs = idcs.sort(0)
    logits= logits[odidcs]
    if to_cpu:
        logits, idcs_sorted = [tsr.to('cpu') for tsr in (logits, idcs_sorted)]
    return logits, idcs_sorted

from typing import Tuple
def teacher_student_logits_idcs(class_dataset_withidx, trans, teacher:nn.Module, model:nn.Module, device, bs:int=200, to_cpu:bool=False):
    loader = DataLoader(class_dataset_withidx, bs, False, num_workers=2, pin_memory=True)
    t_logits = []
    logits = []
    idcs = []
    with torch.no_grad():
        for tup in loader:
            img, tgt, idx = (tsr.to(device) for tsr in img_tgt_idx(tup))
            img = trans(img)
            t_logits.append(teacher(img))
            logits.append(model(img))
            idcs.append(idx.to(torch.long))
        logits = torch.cat(logits, 0)
        t_logits = torch.cat(t_logits, 0)
        idcs = torch.cat(idcs, 0)
        # print('logits', logits)
        # print('t_logits', t_logits)
    idcs_sorted, odidcs = idcs.sort(0)
    logits= logits[odidcs]
    t_logits = t_logits[odidcs]
    if to_cpu:
        t_logits, logits, idcs_sorted = [tsr.to('cpu') for tsr in (t_logits, logits, idcs_sorted)]
    return t_logits, logits, idcs_sorted

from torch import Tensor as Tsr
import torch.nn.functional as F
def compute_clarity(t_preds:Tsr, logits:Tsr, softmax:bool=True, device=None):
    if device is not None:
        t_preds, logits = [tsr.to(device) for tsr in (t_preds, logits)]
    if softmax:
        logits = F.softmax(logits, 1)
    s_t_pred = (logits * F.one_hot(t_preds, num_classes=logits.size(1))).sum(dim=1)
    s_pred_2nd = (logits - F.one_hot(t_preds, num_classes=logits.size(1))*99999999).max(dim=1)[0]
    return s_t_pred - s_pred_2nd

def compute_negloss(t_logits, logits, use_kl:bool=True, device=None):
    if device is not None:
        t_logits, logits = [tsr.to(device) for tsr in (t_logits, logits)]
    if not use_kl:
        t_logits, logits = [F.softmax(tsr, 1) for tsr in (t_logits, logits)]
        ce = nn.CrossEntropyLoss(reduction='none')
        negloss = -ce(t_logits, logits)
    else:
        kl = nn.KLDivLoss(reduction='none')
        targets = F.softmax(t_logits/20.0, dim=1)
        outs = F.log_softmax(logits/20.0, dim=1)
        negloss = -kl(outs, targets)
        # print('negloss', negloss)
    return negloss.sum(dim=1)

import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
import time
from torchvision import transforms
def train_epoch(size:int,
                train_loader:DataLoader,
                nclass:int,
                model:nn.Module,
                optimizer:Optimizer,
                device,
                mixup='vanilla',
                cutmix_tup=None,
                trans=None,
                ):
    if mixup == 'cut':
        assert cutmix_tup is not None
        mix_p, beta = cutmix_tup
    elif mixup == 'vanilla':
        mix_p = 0
    else:
        raise NotImplementedError

    if trans is None:
        trans = transforms.Compose([])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    num_gpu = torch.cuda.device_count()
    rsz = transforms.Resize(size)

    model.train()
    end = time.time()
    num_exp = 0
    for i, (input, target) in enumerate(train_loader):
        # if train_loader.device == 'cpu':
        if input.size(0) < num_gpu:
            continue
        input = input.to(device)
        target = target.to(device)

        input = trans(input)
        data_time.update(time.time() - end)

        r = np.random.rand(1)
        if r < mix_p:
            # generate mixed sample
            with torch.no_grad():
                lam = np.random.beta(beta, beta)
                rand_index = random_indices(target, nclass=nclass)

                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                input = rsz(input)
            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            # compute output
            output = model(rsz(input))
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_exp += len(target)

    # if (epoch % args.epoch_print_freq == 0) and (logger is not None) and args.verbose == True:
    #     logger(
    #         '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
    #         .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg

def teach_epoch(
                teacher:nn.Module,
                size:int,
                train_loader:DataLoader,
                nclass:int,
                model:nn.Module,
                optimizer:Optimizer,
                device,
                mixup='vanilla',
                cutmix_tup=None,
                trans=None,
                ):
    if mixup == 'cut':
        assert cutmix_tup is not None
        mix_p, beta = cutmix_tup
    elif mixup == 'vanilla':
        mix_p = 0
    else:
        raise NotImplementedError
    if trans is None:
        trans = transforms.Compose([])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    kl = nn.KLDivLoss(reduction='batchmean')
    kl.to(device)
    rsz = transforms.Resize(size)

    model.train()
    end = time.time()
    num_exp = 0
    for i, (pinput, ptarget) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        pinput, ptarget = [tsr.to(device) for tsr in [pinput, ptarget]]
        pinput = trans(pinput)
        data_time.update(time.time() - end)
        r = np.random.rand(1)
        img = pinput
        tgt = ptarget
        if r < mix_p:
            with torch.no_grad():
                lam = np.random.beta(beta, beta)
                rand_index = random_indices(tgt, nclass=nclass)
                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                imgperm = torch.randperm(img.size(0), device=device)
                img = img[imgperm]
                target_t = F.softmax(teacher(img)/20.0, dim=1)
                img = rsz(img)
            output = F.log_softmax(model(img)/20.0, dim=1)
            losskl = kl(output, target_t)
        else:
            with torch.no_grad():
                target_t = F.softmax(teacher(img)/20.0, dim=1)
                img = rsz(img)
            output = F.log_softmax(model(img)/20.0, dim=1)
            losskl = kl(output, target_t)
        loss = losskl

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), ptarget.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # num_exp += ntarget.size(0)
        # if (n_data > 0) and (num_exp >= n_data):
        #     break

    # if (epoch % args.epoch_print_freq == 0) and (logger is not None) and args.verbose == True:
    #     logger(
    #         '(Train) [Epoch {0}/{1}] {2}  Loss {loss.avg:.3f}'
    #         .format(epoch, args.epochs, get_time(), loss=losses))

    return top1.avg, top5.avg, losses.avg

def validate(size:int, val_loader:DataLoader, model:nn.Module, epoch:int, device, logger=None, trans=None):
    if trans is None:
        trans = transforms.Compose([])
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_gpu = torch.cuda.device_count()
    criterion = nn.CrossEntropyLoss()
    rsz = transforms.Resize(size)
    if logger is None:
        logger = print

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if input.size(0) < num_gpu:
            model_ = model.module
        else:
            model_ = model
        input = input.to(device)
        input = trans(input)
        target = target.to(device)
        output = model_(rsz(input))

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger('(Test ) [Epoch {0}] {1} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, get_time(), top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg

from typing import Iterable, Callable, Union
from tqdm import tqdm
def train(
        size:int, 
        model:nn.Module, 
        train_loader:DataLoader,
        val_loader:DataLoader, 
        nclass:int,
        opt_getter:Callable,
        scheduler_getter:Callable,
        epochs:int,
        device,
        logger=None, 
        train_trans=None, 
        val_trans=None, 
        mixup='vanilla',
        cutmix_tup=None,
        default_bs:int=64,
        valid=True, lr_mul=1.0, use_sche=True, print_freq:int=10, save_best=False, start_eval=0.0, 
        teacher=None):
    if valid:
        assert val_loader is not None
    if logger is None:
        logger = print
    if train_trans is None:
        train_trans = transforms.Compose([])
    if val_trans is None:
        val_trans = transforms.Compose([])
    # criterion = nn.CrossEntropyLoss().to(device)
    lr_mul = lr_mul * train_loader.batch_size/default_bs
    if teacher is None:
        epoch_func = train_epoch
    else:
        epoch_func = teach_epoch
    
    optimizer:Optimizer = opt_getter(model.parameters(), lr_mul)
    print('optimizer: ', type(optimizer))
    scheduler = scheduler_getter(optimizer, epochs)
    print('scheduler: ', type(scheduler))
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0

    logger(f"Start training with base augmentation and {mixup} mixup")

    # Start training and validation
    rng = range(cur_epoch + 1, epochs + 1)
    epchin = [size, train_loader]
    logger(f'{len(epchin)-1} dataloaders in total')
    if teacher is not None:
        epchin = [teacher, *epchin]
    epchin += [nclass, model, optimizer, device, mixup, cutmix_tup, train_trans]
    logger(f'cutmix tup: {cutmix_tup[0]}, {cutmix_tup[1]}')
    for epoch in rng:
        acc1_tr, _, loss_tr = epoch_func(*epchin)
        if epoch % print_freq == 0:
            logger(
            '(Train) [Epoch {0}/{1}] {2}  Acc1 {acc1:.3f}, Loss {loss:.3f}'
            .format(epoch, epochs, get_time(), acc1=acc1_tr, loss=loss_tr))
        if epoch % print_freq == 0 and epoch >= epochs*start_eval:
            with torch.no_grad():
                #validation set
                if valid:
                    acc1, acc5, loss_val = validate(size, val_loader, model, epoch, device, logger, trans=val_trans)
                is_best = acc1 > best_acc1
                if is_best:
                    best_acc1 = acc1
                    best_acc5 = acc5
                    logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')
                    try:
                        best_stdt = {k:v.detach().clone() for k,v in model.module.state_dict().items()}
                    except:
                        best_stdt = {k:v.detach().clone() for k,v in model.state_dict().items()}

        if use_sche:
            scheduler.step()
    if save_best:
        model.module.load_state_dict(best_stdt)

    return best_acc1, acc1