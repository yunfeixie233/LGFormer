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
        classes=('Background', 'Aeroplane_Body', 'Aeroplane_Engine', 'Aeroplane_Wing',
                'Aeroplane_Stern', 'Aeroplane_Wheel', 'Bicycle_Wheel', 'Bicycle_Body',
                'Bird_Head', 'Bird_Wing', 'Bird_Leg', 'Bird_Torso', 'Boat', 
                'Bottle_Cap', 'Bottle_Body', 'Bus_Window', 'Bus_Wheel', 'Bus_Body',
                'Car_Window', 'Car_Wheel', 'Car_Light', 'Car_Plate', 'Car_Body',
                'Cat_Head', 'Cat_Leg', 'Cat_Tail', 'Cat_Torso', 'Chair', 'Cow_Head',
                'Cow_Tail', 'Cow_Leg', 'Cow_Torso', 'Dining_Table', 'Dog_Head',
                'Dog_Leg', 'Dog_Tail', 'Dog_Torso', 'Horse_Head', 'Horse_Tail',
                'Horse_Leg', 'Horse_Torso', 'Motorbike_Wheel', 'Motorbike_Body',
                'Person_Head', 'Person_Torso', 'Person_Lower_Arm', 'Person_Upper_Arm',
                'Person_Lower_Leg', 'Person_Upper_Leg', 'Potted_Plant_Pot',
                'Potted_Plant_Plant', 'Sheep_Head', 'Sheep_Leg', 'Sheep_Torso',
                'Sofa', 'Train', 'TV_Screen', 'TV_Frame'))

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
