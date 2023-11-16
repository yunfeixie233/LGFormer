# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PartImagenetDataset(BaseSegDataset):
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
                 [0, 0, 0]],
            )

    def __init__(self,
                 img_suffix='.JPEG',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
