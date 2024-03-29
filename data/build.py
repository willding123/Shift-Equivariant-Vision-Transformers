# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import Mixup
from timm.data import create_transform
import torch

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler
# from torch.distributed.elastic.multiprocessing.errors import record


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {0} / global rank {0} successfully build train dataset")

    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config, roll=False)
    print(f"local rank {0} / global rank {0} successfully build val dataset")
    dataset_val_adversarial, _ = build_dataset(is_train=False, config=config, roll=True)

    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    # num_tasks = 1
    global_rank = dist.get_rank()
    # global_rank = 0
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        # indices = np.arange(0, len(dataset_train), 1)
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_val_adversarial = torch.utils.data.SequentialSampler(dataset_val_adversarial)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )
        sampler_val_adversarial = torch.utils.data.distributed.DistributedSampler(
            dataset_val_adversarial, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    
    data_loader_val_adversarial = torch.utils.data.DataLoader(
        dataset_val_adversarial, sampler=sampler_val_adversarial,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, data_loader_val_adversarial, mixup_fn


def build_dataset(is_train, config, roll=False):
    transform = build_transform(is_train, config, roll=roll)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        # if roll:
        #     if config.DATA.SHIFT_MAX > 0:
        #         prefix = prefix + '_shifted' + str(config.DATA.SHIFT_MAX)
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config, roll = False):
    resize_im = config.DATA.IMG_SIZE > 32
    shift_roll = config.DATA.SHIFT_MAX > 0
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.DATA.MEAN,
            std=config.DATA.STD
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        # if shift_roll and roll:
        #     # apply torch.roll in range [-shift_max, shift_max]
        #     if abs(config.DATA.SHIFT_SIZE) > 0 :
        #         print("shifted by ", config.DATA.SHIFT_SIZE, " pixels")
        #         transform.transforms.append(transforms.Lambda(lambda x: shift_utils(x, config.DATA.SHIFT_SIZE)))
        #     else:
        #         transform.transforms.append(transforms.Lambda(lambda x: torch.roll(x, shifts=( np.random.randint(-config.DATA.SHIFT_MAX, config.DATA.SHIFT_MAX), np.random.randint(-config.DATA.SHIFT_MAX, config.DATA.SHIFT_MAX)), dims=(1, 2))))
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.DATA.MEAN, config.DATA.STD))
    if shift_roll and roll:
        # apply torch.roll in range [-shift_max, shift_max]
        if abs(config.DATA.SHIFT_SIZE) > 0 :
            # print("shifted by ", config.DATA.SHIFT_SIZE, " pixels")
            t.append(transforms.Lambda(lambda x: torch.roll(x, (config.DATA.SHIFT_SIZE, config.DATA.SHIFT_SIZE), (1, 2))))
        else:
            # t.append(transforms.Lambda(lambda x: torch.roll(x, shifts=( np.random.randint(-config.DATA.SHIFT_MAX, config.DATA.SHIFT_MAX), np.random.randint(-config.DATA.SHIFT_MAX, config.DATA.SHIFT_MAX)), dims=(0, 1))))
            # generate a two-element tuple of random integers in the range [0, config.DATA.SHIFT_MAX)
            # shifts = (np.random.randint(-config.DATA.SHIFT_MAX, config.DATA.SHIFT_MAX), np.random.randint(-config.DATA.SHIFT_MAX, config.DATA.SHIFT_MAX))
            # t.append(transforms.Lambda(lambda x: torch.roll(x, shifts=shifts, dims=(1,2))))
            t.append(transforms.RandomAffine(degrees=0, translate=(config.DATA.SHIFT_MAX/config.DATA.IMG_SIZE, config.DATA.SHIFT_MAX/config.DATA.IMG_SIZE), scale=None, shear=None, resample=False, fill=0))

    return transforms.Compose(t)
