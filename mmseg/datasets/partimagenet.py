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
            ))

    def __init__(self,
                 img_suffix='.JPEG',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
