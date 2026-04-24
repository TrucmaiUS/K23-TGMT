import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .bases import ImageDataset
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


def _swap_flipped_semantic_labels(mask_array, flip_label_pairs):
    if not flip_label_pairs:
        return mask_array

    swapped_mask = mask_array.copy()
    for left_label, right_label in flip_label_pairs:
        left_region = mask_array == left_label
        right_region = mask_array == right_label
        swapped_mask[left_region] = right_label
        swapped_mask[right_region] = left_label
    return swapped_mask


class ReIDPairTransform(object):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.size = cfg.INPUT.SIZE_TRAIN if is_train else cfg.INPUT.SIZE_TEST
        self.flip_prob = cfg.INPUT.PROB
        self.padding = cfg.INPUT.PADDING
        self.pixel_mean = cfg.INPUT.PIXEL_MEAN
        self.pixel_std = cfg.INPUT.PIXEL_STD
        self.re_prob = cfg.INPUT.RE_PROB if is_train else 0.0
        self.image_interp = _bicubic_mode()
        self.mask_interp = _nearest_mode()
        self.flip_label_pairs = [
            (int(pair[0]), int(pair[1]))
            for pair in cfg.MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS
            if len(pair) == 2
        ]

    def _apply_paired_random_erasing(self, img, semantic_mask=None):
        if self.re_prob <= 0.0 or torch.rand(1).item() >= self.re_prob:
            return img, semantic_mask

        area = img.shape[1] * img.shape[2]
        for _ in range(10):
            target_area = torch.empty(1).uniform_(0.02, 0.4).item() * area
            aspect_ratio = torch.empty(1).uniform_(0.3, 3.3).item()

            erase_h = int(round((target_area * aspect_ratio) ** 0.5))
            erase_w = int(round((target_area / max(aspect_ratio, 1e-6)) ** 0.5))

            if erase_h < img.shape[1] and erase_w < img.shape[2]:
                top = torch.randint(0, img.shape[1] - erase_h + 1, (1,)).item()
                left = torch.randint(0, img.shape[2] - erase_w + 1, (1,)).item()
                img[:, top:top + erase_h, left:left + erase_w] = torch.empty(
                    img.shape[0], erase_h, erase_w, dtype=img.dtype
                ).normal_()
                if semantic_mask is not None:
                    semantic_mask[top:top + erase_h, left:left + erase_w] = 0
                return img, semantic_mask

        return img, semantic_mask

    def __call__(self, img, semantic_mask=None):
        img = TF.resize(img, self.size, interpolation=self.image_interp)
        if semantic_mask is not None:
            semantic_mask = TF.resize(semantic_mask, self.size, interpolation=self.mask_interp)

        if self.is_train:
            if torch.rand(1).item() < self.flip_prob:
                img = TF.hflip(img)
                if semantic_mask is not None:
                    semantic_mask = TF.hflip(semantic_mask)
                    if self.flip_label_pairs:
                        flipped_mask = np.array(semantic_mask, dtype=np.uint8)
                        flipped_mask = _swap_flipped_semantic_labels(flipped_mask, self.flip_label_pairs)
                        semantic_mask = Image.fromarray(flipped_mask, mode='L')

            img = TF.pad(img, self.padding)
            if semantic_mask is not None:
                semantic_mask = TF.pad(semantic_mask, self.padding, fill=0)

            top, left, height, width = T.RandomCrop.get_params(img, output_size=self.size)
            img = TF.crop(img, top, left, height, width)
            if semantic_mask is not None:
                semantic_mask = TF.crop(semantic_mask, top, left, height, width)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.pixel_mean, std=self.pixel_std)

        if semantic_mask is None:
            img, _ = self._apply_paired_random_erasing(img, None)
            return img, None

        semantic_mask = torch.from_numpy(np.array(semantic_mask, dtype=np.int64))
        img, semantic_mask = self._apply_paired_random_erasing(img, semantic_mask)
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
