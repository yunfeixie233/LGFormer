# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine

@DATASETS.register_module()
class PascalVOCDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)

@DATASETS.register_module()
class PascalPartDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
classes = (
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
    'object 51', 'object 52', 'object 53', 'object 54', )) 

        # palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        #          [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        #          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        #          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        #          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        #          [0, 64, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class PascalPartDataset_2(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
    classes = (
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
        'object 51', 'object 52', 'object 53', 'object 54','object 55',)) 

    palette=[
        [120, 120, 120], [172, 122, 121], [104, 228, 228], [211, 21, 242], [36, 87, 70],
        [216, 88, 140], [45, 75, 171], [171, 51, 218], [157, 71, 193], [25, 77, 72],
        [9, 148, 115], [208, 243, 197], [254, 79, 175], [192, 82, 99], [216, 177, 243],
        [29, 147, 147], [142, 167, 32], [193, 9, 185], [127, 32, 31], [188, 79, 33],
        [42, 98, 204], [120, 222, 231], [34, 128, 128], [164, 53, 133], [38, 232, 244],
        [17, 79, 132], [105, 42, 186], [31, 120, 1], [65, 231, 169], [57, 35, 102],
        [117, 88, 206], [117, 251, 220], [244, 188, 65], [118, 251, 101], [231, 64, 43],
        [115, 243, 252], [241, 253, 90], [93, 19, 242], [232, 78, 143], [148, 227, 186],
        [23, 207, 141], [82, 151, 141], [155, 199, 228], [227, 197, 118], [215, 118, 88],
        [113, 178, 36], [162, 48, 93], [131, 98, 42], [205, 112, 231], [149, 201, 127],
        [0, 138, 114], [43, 186, 127], [23, 187, 130], [121, 98, 62], [163, 222, 123],
        [195, 82, 174],
]


    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
        
@DATASETS.register_module()
class PascalPartDataset_obj(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
classes = (
    'object 1', 'object 2', 'object 3', 'object 4', 'object 5', 
    'object 6', 'object 7', 'object 8', 'object 9', 'object 10', 
    'object 11', 'object 12', 'object 13', 'object 14', 'object 15', 
    'object 16', 'object 17',)) 

        # palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        #          [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        #          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        #          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        #          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        #          [0, 64, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)        
        
        
@DATASETS.register_module()
class PascalPartDataset_joint(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes_obj = (
            'object 1', 'object 2', 'object 3', 'object 4', 'object 5', 
            'object 6', 'object 7', 'object 8', 'object 9', 'object 10', 
            'object 11', 'object 12', 'object 13', 'object 14', 'object 15', 
            'object 16', 'object 17',) ,
        classes_part = (
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
        'object 51', 'object 52', 'object 53', 'object 54','object 55',))
    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
       

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
                        