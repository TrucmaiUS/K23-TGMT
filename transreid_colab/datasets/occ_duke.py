# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class OCC_DukeMTMCreID(BaseImageDataset):
    """
    Occluded-DukeMTMC
    Reference:
    1. Miao et al. Pose-Guided Feature Alignment for Occluded Person Re-Identification. ICCV 2019.
    2. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    Split lists: https://github.com/lightas/Occluded-DukeMTMC-Dataset

    Dataset statistics:
    # identities: 702 train + 519 query
    # images: 15618 train + 2210 query + 17661 gallery
    # cameras: 8
    """
    dataset_dir = 'Occluded_Duke'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(OCC_DukeMTMCreID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.pid_begin = pid_begin
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Occluded-DukeMTMC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(
                "'{}' is not available. Run tools/prepare_occluded_duke.py first.".format(self.dataset_dir)
            )
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        cam_container = set()
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
            cam_container.add(camid)
        return dataset
