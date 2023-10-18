# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType

import os
import torch
import os.path as osp
import h5py
def to_h5(output_directory, max_keys_per_file=None, **kwargs):
    """
    Save input tensors or numpy arrays to an H5 file in a specified directory with automatic numbering.
    Each key from kwargs gets its own sub-directory.
    
    Usage:
    >>> a = torch.tensor([1,2,3])
    >>> b = np.array([4,5,6])
    >>> to_h5(output_directory='./output', a=a, b=b)
    
    Arguments:
    output_directory: The main directory where the sub-directories and H5 files should be saved.
    max_keys_per_file: Maximum number of keys (datasets) allowed in each H5 file.
    **kwargs : Tensors or numpy arrays to save.
    
    Returns:
    None
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Helper function to handle data writing
    def write_data(f, key, value):
        if torch.is_tensor(value) and value.device.type == 'cuda':
            value = value.detach().cpu().numpy()
        elif torch.is_tensor(value):
            value = value.detach().numpy()
        f.create_dataset(key, data=value)
    
    for key, value in kwargs.items():
        sub_directory = osp.join(output_directory, key)
        
        if not os.path.exists(sub_directory):
            os.makedirs(sub_directory)
        
        existing_files = [f for f in os.listdir(sub_directory) if f.startswith(key) and f.endswith('.h5')]
        counts = [int(f.split(key)[1].split('.h5')[0]) for f in existing_files]
        max_count = max(counts, default=0)  # get the maximum count or set it to 0 if the folder is empty
        
        output_file = osp.join(sub_directory, f'{key}{max_count}.h5')
        
        with h5py.File(output_file, "a") as f:
            keys_in_file = list(f.keys())
            
            if max_keys_per_file is not None and len(keys_in_file) >= max_keys_per_file:
                max_count += 1
                output_file = osp.join(sub_directory, f'{key}{max_count}.h5')
                f.close()
                f = h5py.File(output_file, "a")
                
            write_data(f, key, value)
            f.close()

@MODELS.register_module()
class SimpleFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        vis = False
        if vis:
            import h5py
            to_h5('/data2/yunfei/vis',max_keys_per_file=1,input = input)
        inputs = []            
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        if vis:
            for out in outs:
                import h5py
                to_h5('/data2/yunfei/vis',max_keys_per_file=1,out = out)
        return tuple(outs)
