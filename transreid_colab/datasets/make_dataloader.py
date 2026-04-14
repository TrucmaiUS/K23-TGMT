import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
}


def _bicubic_mode():
    if hasattr(T, 'InterpolationMode'):
        return T.InterpolationMode.BICUBIC
    return Image.BICUBIC


def _nearest_mode():
    if hasattr(T, 'InterpolationMode'):
        return T.InterpolationMode.NEAREST
    return Image.NEAREST


class ReIDPairTransform(object):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.size = cfg.INPUT.SIZE_TRAIN if is_train else cfg.INPUT.SIZE_TEST
        self.flip_prob = cfg.INPUT.PROB
        self.padding = cfg.INPUT.PADDING
        self.pixel_mean = cfg.INPUT.PIXEL_MEAN
        self.pixel_std = cfg.INPUT.PIXEL_STD
        self.random_erasing = RandomErasing(
            probability=cfg.INPUT.RE_PROB,
            mode='pixel',
            max_count=1,
            device='cpu'
        ) if is_train else None
        self.image_interp = _bicubic_mode()
        self.mask_interp = _nearest_mode()

    def __call__(self, img, semantic_mask=None):
        img = TF.resize(img, self.size, interpolation=self.image_interp)
        if semantic_mask is not None:
            semantic_mask = TF.resize(semantic_mask, self.size, interpolation=self.mask_interp)

        if self.is_train:
            if torch.rand(1).item() < self.flip_prob:
                img = TF.hflip(img)
                if semantic_mask is not None:
                    semantic_mask = TF.hflip(semantic_mask)

            img = TF.pad(img, self.padding)
            if semantic_mask is not None:
                semantic_mask = TF.pad(semantic_mask, self.padding, fill=0)

            top, left, height, width = T.RandomCrop.get_params(img, output_size=self.size)
            img = TF.crop(img, top, left, height, width)
            if semantic_mask is not None:
                semantic_mask = TF.crop(semantic_mask, top, left, height, width)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.pixel_mean, std=self.pixel_std)

        if self.random_erasing is not None:
            img = self.random_erasing(img)

        if semantic_mask is None:
            return img, None

        semantic_mask = torch.from_numpy(np.array(semantic_mask, dtype=np.int64))
        return img, semantic_mask


def train_collate_fn(batch):
    if len(batch[0]) == 6:
        imgs, pids, camids, viewids, _, semantic_masks = zip(*batch)
    else:
        imgs, pids, camids, viewids, _ = zip(*batch)
        semantic_masks = None

    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    if semantic_masks is not None:
        semantic_masks = torch.stack(semantic_masks, dim=0)
        return torch.stack(imgs, dim=0), pids, camids, viewids, semantic_masks

    return torch.stack(imgs, dim=0), pids, camids, viewids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = ReIDPairTransform(cfg, is_train=True)
    val_transforms = ReIDPairTransform(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    semantic_cfg = None
    if cfg.MODEL.SEM_ALIGN.ENABLED:
        semantic_cfg = {
            "mask_dir": cfg.MODEL.SEM_ALIGN.MASK_DIR,
            "mask_ext": cfg.MODEL.SEM_ALIGN.MASK_EXT,
        }

    train_set = ImageDataset(
        dataset.train,
        train_transforms,
        dataset_root=dataset.dataset_dir,
        semantic_cfg=semantic_cfg
    )
    train_set_normal = ImageDataset(
        dataset.train,
        val_transforms,
        dataset_root=dataset.dataset_dir
    )
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(
        dataset.query + dataset.gallery,
        val_transforms,
        dataset_root=dataset.dataset_dir
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
