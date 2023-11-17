# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


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