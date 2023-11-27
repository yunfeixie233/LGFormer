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
            classes_part = (
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
        palette_part=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [0, 0, 0]
                 ],           
    classes_obj = (
        'object 1', 'object 2', 'object 3', 'object 4', 'object 5', 
        'object 6', 'object 7', 'object 8', 'object 9', 'object 10', 
        'object 11', 'object 12', 'object 13', 'object 14', 'object 15', 
        'object 16', 'object 17', 'object 18', 'object 19', 'object 20', 
        'object 21', 'object 22', 'object 23', 'object 24', 'object 25', 
        'object 26', 'object 27', 'object 28', 'object 29', 'object 30', 
        'object 31', 'object 32', 'object 33', 'object 34', 'object 35', 
        'object 36', 'object 37', 'object 38', 'object 39', 'object 40', 
        'object 41', 'object 42', 'object 43', 'object 44', 'object 45', 
        'object 46', 'object 47', 'object 48', 'object 49', 'object 50', 
        'object 51', 'object 52', 'object 53', 'object 54', 'object 55', 
        'object 56', 'object 57', 'object 58', 'object 59', 'object 60', 
        'object 61', 'object 62', 'object 63', 'object 64', 'object 65', 
        'object 66', 'object 67', 'object 68', 'object 69', 'object 70', 
        'object 71', 'object 72', 'object 73', 'object 74', 'object 75', 
        'object 76', 'object 77', 'object 78', 'object 79', 'object 80', 
        'object 81', 'object 82', 'object 83', 'object 84', 'object 85', 
        'object 86', 'object 87', 'object 88', 'object 89', 'object 90', 
        'object 91', 'object 92', 'object 93', 'object 94', 'object 95', 
        'object 96', 'object 97', 'object 98', 'object 99', 'object 100', 
        'object 101', 'object 102', 'object 103', 'object 104', 'object 105', 
        'object 106', 'object 107', 'object 108', 'object 109', 'object 110', 
        'object 111', 'object 112', 'object 113', 'object 114', 'object 115', 
        'object 116', 'object 117', 'object 118', 'object 119', 'object 120', 
        'object 121', 'object 122', 'object 123', 'object 124', 'object 125', 
        'object 126', 'object 127', 'object 128', 'object 129', 'object 130', 
        'object 131', 'object 132', 'object 133', 'object 134', 'object 135', 
        'object 136', 'object 137', 'object 138', 'object 139', 'object 140', 
        'object 141', 'object 142', 'object 143', 'object 144', 'object 145', 
        'object 146', 'object 147', 'object 148', 'object 149', 'object 150', 
        'object 151', 'object 152', 'object 153', 'object 154', 'object 155', 
        'object 156', 'object 157', 'object 158','background'
    ),
    palette_obj=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], 
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], 
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], 
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [105, 105, 105], [165, 105, 105], [0, 215, 215], 
                 [65, 35, 35], [0, 185, 18], [105, 105, 65], [125, 125, 140], 
                 [189, 0, 240], [215, 215, 245], [0, 235, 22], [209, 20, 240], 
                 [220, 240, 22], [135, 0, 46], [105, 135, 55], [0, 240, 66], 
                 [240, 21, 67], [128, 240, 125], [189, 240, 0], [240, 36, 0],
                 [189, 55, 0], [0, 87, 185], [46, 215, 235], [240, 0, 36], 
                 [0, 87, 240], [240, 0, 209], [0, 0, 215], [205, 205, 205],
                 [240, 0, 107], [97, 24, 240], [0, 240, 199], [22, 240, 209], 
                 [240, 169, 0], [10, 240, 86], [240, 26, 25], [22, 240, 240], 
                 [87, 0, 240], [240, 61, 21], [240, 179, 22], [0, 15, 15], 
                 [105, 105, 135], [165, 105, 135], [0, 215, 245], [65, 35, 65],
                 [19, 185, 0], [105, 135, 95], [125, 140, 125], [189, 20, 255],
                 [215, 245, 215], [19, 235, 0], [239, 0, 240], [250, 240, 0],
                 [135, 0, 76], [135, 105, 55], [23, 240, 36], [240, 21, 97], 
                 [128, 240, 155], [189, 255, 19], [189, 70, 18], [0, 117, 185],
                 [46, 245, 235], [255, 21, 36], [0, 117, 240], [240, 24, 224], 
                 [0, 0, 245], [205, 205, 235], [127, 0, 240], [25, 240, 71],
                 [240, 76, 0], [240, 209, 0], [15, 0, 15], [90, 90, 90],
                 [150, 90, 90], [0, 200, 200], [50, 20, 20], [0, 170, 3],
                 [90, 90, 50], [110, 110, 155], [174, 0, 225], [200, 200, 255],
                 [0, 220, 37], [194, 20, 225], [205, 225, 7], [120, 0, 31], 
                 [90, 120, 40], [0, 225, 81], [225, 6, 52], [113, 225, 110],
                 [174, 225, 0], [225, 21, 0], [174, 40, 0], [0, 72, 170],
                 [31, 200, 220], [225, 0, 21], [0, 72, 225], [225, 0, 194], 
                 [0, 15, 200], [190, 190, 190], [225, 0, 92], [82, 24, 225],
                 [0, 225, 184], [22, 225, 194], [225, 154, 0], [0, 255, 101],
                 [225, 26, 40], [22, 225, 255], [72, 0, 225], [225, 46, 6], 
                 [225, 164, 22], [0, 0, 30], [105, 135, 135], [165, 135, 105], 
                 [21, 215, 230], [65, 65, 35], [19, 200, 18], [135, 105, 95], 
                 [125, 140, 155], [215, 245, 245], [190, 175, 35],[0, 0, 0],]
    
    )
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
