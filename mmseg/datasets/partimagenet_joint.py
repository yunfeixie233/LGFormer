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
