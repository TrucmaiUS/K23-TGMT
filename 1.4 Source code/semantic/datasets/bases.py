from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_semantic_mask(mask_path):
    got_mask = False
    if not osp.exists(mask_path):
        raise IOError("{} does not exist".format(mask_path))
    while not got_mask:
        try:
            semantic_mask = Image.open(mask_path)
            got_mask = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(mask_path))
            pass
    return semantic_mask


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, dataset_root=None, semantic_cfg=None):
        self.dataset = dataset
        self.transform = transform
        self.dataset_root = dataset_root
        self.semantic_cfg = semantic_cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        semantic_mask = None

        if self.semantic_cfg is not None:
            semantic_mask = read_semantic_mask(self._get_semantic_mask_path(img_path))

        if self.transform is not None:
            try:
                img, semantic_mask = self.transform(img, semantic_mask)
            except TypeError:
                img = self.transform(img)

        if semantic_mask is not None:
            return img, pid, camid, trackid, img_path.split('/')[-1], semantic_mask

        return img, pid, camid, trackid, img_path.split('/')[-1]

    def _get_semantic_mask_path(self, img_path):
        if self.dataset_root is None:
            raise ValueError('dataset_root is required when semantic alignment is enabled.')

        relative_path = osp.relpath(img_path, self.dataset_root)
        stem, _ = osp.splitext(relative_path)
        return osp.join(
            self.dataset_root,
            self.semantic_cfg["mask_dir"],
            stem + self.semantic_cfg["mask_ext"]
        )
