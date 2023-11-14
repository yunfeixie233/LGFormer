# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS

@DATASETS.register_module()
class PartImagenetDataset_joint(BaseSegDataset):
    """PARTIMAGENET dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
            classes = (
                'Quadruped Head',
                'Quadruped Body',
                'Quadruped Foot',
                'Quadruped Tail',
                'Biped Head',
                'Biped Body',
                'Biped Hand',
                'Biped Foot',
                'Biped Tail',
                'Fish Head',
                'Fish Body',
                'Fish Fin',
                'Fish Tail',
                'Bird Head',
                'Bird Body',
                'Bird Wing',
                'Bird Foot',
                'Bird Tail',
                'Snake Head',
                'Snake Body',
                'Reptile Head',
                'Reptile Body',
                'Reptile Foot',
                'Reptile Tail',
                'Car Body',
                'Car Tier',
                'Car Side Mirror',
                'Bicycle Body',
                'Bicycle Head',
                'Bicycle Seat',
                'Bicycle Tier',
                'Boat Body',
                'Boat Sail',
                'Aeroplane Head',
                'Aeroplane Body',
                'Aeroplane Engine',
                'Aeroplane Wing',
                'Aeroplane Tail',
                'Bottle Mouth',
                'Bottle Body',
                'background'
            ),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8]],
            )
# classes = (
#     'object 1', 'object 2', 'object 3', 'object 4', 'object 5', 
#     'object 6', 'object 7', 'object 8', 'object 9', 'object 10', 
#     'object 11', 'object 12', 'object 13', 'object 14', 'object 15', 
#     'object 16', 'object 17', 'object 18', 'object 19', 'object 20', 
#     'object 21', 'object 22', 'object 23', 'object 24', 'object 25', 
#     'object 26', 'object 27', 'object 28', 'object 29', 'object 30', 
#     'object 31', 'object 32', 'object 33', 'object 34', 'object 35', 
#     'object 36', 'object 37', 'object 38', 'object 39', 'object 40', 
#     'object 41', 'object 42', 'object 43', 'object 44', 'object 45', 
#     'object 46', 'object 47', 'object 48', 'object 49', 'object 50', 
#     'object 51', 'object 52', 'object 53', 'object 54', 'object 55', 
#     'object 56', 'object 57', 'object 58', 'object 59', 'object 60', 
#     'object 61', 'object 62', 'object 63', 'object 64', 'object 65', 
#     'object 66', 'object 67', 'object 68', 'object 69', 'object 70', 
#     'object 71', 'object 72', 'object 73', 'object 74', 'object 75', 
#     'object 76', 'object 77', 'object 78', 'object 79', 'object 80', 
#     'object 81', 'object 82', 'object 83', 'object 84', 'object 85', 
#     'object 86', 'object 87', 'object 88', 'object 89', 'object 90', 
#     'object 91', 'object 92', 'object 93', 'object 94', 'object 95', 
#     'object 96', 'object 97', 'object 98', 'object 99', 'object 100', 
#     'object 101', 'object 102', 'object 103', 'object 104', 'object 105', 
#     'object 106', 'object 107', 'object 108', 'object 109', 'object 110', 
#     'object 111', 'object 112', 'object 113', 'object 114', 'object 115', 
#     'object 116', 'object 117', 'object 118', 'object 119', 'object 120', 
#     'object 121', 'object 122', 'object 123', 'object 124', 'object 125', 
#     'object 126', 'object 127', 'object 128', 'object 129', 'object 130', 
#     'object 131', 'object 132', 'object 133', 'object 134', 'object 135', 
#     'object 136', 'object 137', 'object 138', 'object 139', 'object 140', 
#     'object 141', 'object 142', 'object 143', 'object 144', 'object 145', 
#     'object 146', 'object 147', 'object 148', 'object 149', 'object 150', 
#     'object 151', 'object 152', 'object 153', 'object 154', 'object 155', 
#     'object 156', 'object 157', 'object 158','background'
# )
    def __init__(self,
                 img_suffix='.JPEG',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        data_root="/root/autodl-tmp/ADEChallengeData2016/"
        # img_dir = data_root + "images/training" 
        # ann_dir = data_root+ "annotations/training",       

        img_dir = self.data_prefix.get('img_path', None)
        part_ann_dir = self.data_prefix.get('part_map_path', None)
        obj_ann_dir = self.data_prefix.get('obj_map_path', None)
        
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if part_ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['part_map_path'] = osp.join(part_ann_dir, seg_map)
                if obj_ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['obj_map_path'] = osp.join(obj_ann_dir, seg_map)                    
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if part_ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['part_map_path'] = osp.join(part_ann_dir, seg_map)
                if obj_ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['obj_map_path'] = osp.join(obj_ann_dir, seg_map)                    
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
