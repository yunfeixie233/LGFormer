import math
import warnings
from collections import OrderedDict
from functools import partial
from tkinter import N
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence, Tuple, Union
from cv2 import merge

import einops
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.models import layers as timm_layers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block, _cfg
from timm.models.registry import register_model

from ...superformer.superpixel import superpixel_ops, superpixel_transformer as st
from ...superformer.superpixel.dual_path_transformer_ops import Conv2D
from .decode_head import BaseDecodeHead
# from .misc import Result, interpolate_pos_encoding
from ..builder import HEADS
import sys

LayerScale2d = st.LayerScale2d
SKIP_CONFIRM = False
import os
from ...GPViT.mmcls.gpvit_dev.models.utils.attentions import *
def get_similarity_embedding(similarities, patch_embed, scale_factor):
        similarities = prepare_similarities(
            patch_embed, similarities, scale_factor, merge_multihead_similarities=True
        )
        labels = superpixel_ops.compute_hard_association(similarities)

def _update_params(defaults, **kwargs):
    defaults.update(kwargs)
    return defaults


def get_norm_layer_2d(norm_layer):
    layer_func = (
        norm_layer.func if isinstance(norm_layer, partial) else norm_layer.__class__
    )
    if layer_func is nn.LayerNorm:
        return timm_layers.LayerNorm2d
    else:
        raise NotImplementedError()


def make_superformer_cfg():
    cfg = _cfg()
    cfg["first_conv"] = "patch_embed.conv_stem.conv_layers.0.0.weight"
    return cfg


def init_superformer_weights(module: nn.Module, name: str = ""):
    # if "mergelayer" in name:
    #     if isinstance(module, nn.ModuleList):
    #         for sub_module in module:
    #             if hasattr(sub_module,'weight'):
    #                 nn.init.zeros_(sub_module.weight)
    #             if hasattr(sub_module,'bias'):
    #                 if sub_module.bias is not None:
    #                     nn.init.zeros_(sub_module.bias)
    #     else:
    #         if hasattr(module,'weight'):
    #             nn.init.zeros_(module.weight)
    #         if hasattr(module,'bias'):
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    # else:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        print(f"init_weights: {name}")
        module.init_weights()


def create_downsample(
    stride: int,
    in_channels: int,
    out_channels: int,
    norm_layer_2d: Callable,
    pre_norm: bool,
) -> nn.Module:
    if stride != 1 or in_channels != out_channels:
        assert stride in [1, 2], f"stride must be 1 or 2, got {stride}"
        ds_ks = 1 if stride == 1 else 2
        padding = 0

        downsample = nn.Sequential(
            # TODO(meijieru): norm first or not?
            norm_layer_2d(in_channels),
            timm_layers.create_conv2d(
                in_channels,
                out_channels,
                ds_ks,
                stride=stride,
                bias=True,
                padding=padding,
            ),
        )
    else:
        downsample = norm_layer_2d(in_channels) if pre_norm else nn.Identity()
    return downsample


def prepare_similarities(
    patch_embed: st.SuperPixelTokenization,
    similarities: torch.Tensor,
    scale_factor,
    merge_multihead_similarities: bool = True,
) -> torch.Tensor:
    if similarities.dim() == 5:
        if not merge_multihead_similarities:
            raise NotImplementedError(f"TODO(meijieru): {similarities.shape}")

        # For compatibility with old checkpoints.
        _, num_heads, _, _, _ = similarities.shape
        similarities = similarities.sum(1) / math.sqrt(num_heads)

    b, _, h, w = similarities.shape
    sh, sw = patch_embed.superpixel_shape
    ph, pw = h // sh, w // sw

    similarities = similarities.reshape(b, 9, sh, ph, sw, pw)
    if scale_factor != 1:
        similarities = superpixel_ops.resize_similarities(
            similarities,
            scale_factor=scale_factor,
        )
        similarities = superpixel_ops.maskout_boundary(similarities)
    return similarities
def resize_attn_map(attentions, h, w, align_corners=False):
    """

    Args:
        attentions: shape [B, num_head, H*W, groups]
        h:
        w:

    Returns:

        attentions: shape [B, num_head, h, w, groups]


    """
    scale = (h * w // attentions.shape[2])**0.5
    if h > w:
        w_featmap = w // int(np.round(scale))
        h_featmap = attentions.shape[2] // w_featmap
    else:
        h_featmap = h // int(np.round(scale))
        w_featmap = attentions.shape[2] // h_featmap
    assert attentions.shape[
        2] == h_featmap * w_featmap, f'{attentions.shape[2]} = {h_featmap} x {w_featmap}, h={h}, w={w}'

    bs = attentions.shape[0]
    nh = attentions.shape[1]  # number of head
    groups = attentions.shape[3]  # number of group token
    # [bs, nh, h*w, groups] -> [bs*nh, groups, h, w]
    attentions = rearrange(
        attentions, 'bs nh (h w) c -> (bs nh) c h w', bs=bs, nh=nh, h=h_featmap, w=w_featmap, c=groups)
    attentions = F.interpolate(attentions, size=(h, w), mode='bilinear', align_corners=align_corners)
    #  [bs*nh, groups, h, w] -> [bs, nh, h*w, groups]
    attentions = rearrange(attentions, '(bs nh) c h w -> bs nh h w c', bs=bs, nh=nh, h=h, w=w, c=groups)

    return attentions




def build_2d_sincos_position_embedding(patches_resolution, embed_dim, device, temperature=10000. ):
    h = w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    
    pos_dim = embed_dim // 4
    
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    
    # Generate one set of positional embeddings using the grid for width
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    pos_emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)
    
    # Replicate these embeddings for height
    pos_emb_h = pos_emb_w.clone()
    
    # Concatenate to form the final 2D positional embedding
    pos_emb = torch.cat([pos_emb_w, pos_emb_h], dim=1)[None, :, :]
    
    # Uncomment the following line if you don't want to update the positional embeddings during training
    # pos_emb.requires_grad = False
    pos_emb = nn.Parameter(pos_emb).to(device)
    
    return pos_emb





class PixelRefineSepConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size,
        norm_layer_2d,
        act_layer,
        use_shortcut,
        **kwargs,
    ) -> None:
        super().__init__()
        self.sep_conv = timm_layers.separable_conv.SeparableConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            **kwargs,
        )
        self.norm = norm_layer_2d(in_channels)
        self.act = act_layer()
        self.ls = (
            LayerScale2d(in_channels, init_values=1e-5)
            if use_shortcut
            else nn.Identity()
        )
        self.use_shortcut = use_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.sep_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.ls(x)
        return shortcut + x if self.use_shortcut else x



class SuperformerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_channels_sp: Optional[int],
        out_channels: int,
        stride: int,
        sp_stride: int,
        num_heads: int,
        depth: int,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate: Union[float, Sequence[float]] = 0.0,
        norm_layer=nn.LayerNorm,
        norm_layer_2d=timm_layers.LayerNorm2d,
        act_layer=nn.GELU,
        block_fn=Block,
        ls_init_value: Optional[float] = None,
        superpixel_layer: Optional[st.SuperPixelTokenization] = None,
        superpixel_shape: Tuple[int, int] = (-1, -1),
        seg_block_idx: Optional[int] = None,
        sp_embed_method: str = "identity",
        sp_project_method: str = "noact",
        pre_norm_pixel: bool = False,
        pixel_refine_method: str = "identity",
        return_updated_pixel_features: bool = False,
        unflatten_sp_features: bool = False,
        use_pos_embed: bool = False,
        use_cls_token: bool = False,
        no_embed_class: bool = False,
        use_pixel_similarities: bool = False,
        use_middle_pixel_features: bool = False,
        cross_only: bool = False,
        mergelayer = None,
        merge_pos = None,
        similarity_embedding = False,
        group_embedding = False,
        vis_sp_feature: bool = False,
        output_dir: str= None,
    ) -> None:
        super().__init__()

        assert stride == 1

        print(f"superpixel_shape: {superpixel_shape}")

        self.sp_stride = sp_stride
        # [0, seg_block_idx) are the blocks for segmentation
        self.seg_block_idx = seg_block_idx if seg_block_idx is not None else depth

        self.return_updated_pixel_features = return_updated_pixel_features
        self.unflatten_sp_features = unflatten_sp_features
        self.superpixel_shape = superpixel_shape
        self.no_embed_class = no_embed_class

        self.patch_embed = superpixel_layer()
        self.use_pixel_similarities = use_pixel_similarities
        self.use_middle_pixel_features = use_middle_pixel_features
        self.cross_only = cross_only
        self.mergelayer = mergelayer
        self.merge_pos = merge_pos
        self.similarity_embedding = similarity_embedding
        self.group_embedding = group_embedding
        if pre_norm_pixel:
            raise NotImplementedError()

        if use_pixel_similarities:
            warnings.warn(
                "`use_pixel_similarities` is for compatible with old models only"
            )
            # if not SKIP_CONFIRM:
            #     __import__("ipdb").set_trace()

        if (
            isinstance(self.patch_embed, nn.AvgPool2d)
            or self.patch_embed.superpixel_features_init_method != "from_feature"
            or sp_embed_method == "identity"
        ):
            self.sp_transform_last = nn.Identity()
        elif sp_embed_method == "conv3x3_norm":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
                norm_layer_2d(out_channels),
            )
        elif sp_embed_method == "conv1x1_norm":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 1, padding="same"
                ),
                norm_layer_2d(out_channels),
            )
        elif sp_embed_method == "conv3x3_norm_act":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
                norm_layer_2d(out_channels),
                act_layer(),
            )
        elif sp_embed_method == "conv3x3":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
            )
        else:
            raise ValueError(f"Unknown sp_embed_method: {sp_embed_method}")

        # NOTE(meijieru): break non asym
        self.sp_lift = nn.Identity()
        # if in_channels == out_channels or in_channels_sp is None:
        #     self.sp_lift = nn.Identity()
        # else:
        #     self.sp_lift = nn.Sequential(
        #         timm_layers.create_conv2d(in_channels, out_channels, 1, padding="same"),
        #         norm_layer_2d(out_channels),
        #         act_layer(),
        #     )

        if return_updated_pixel_features:
            assert sp_project_method in ["noact", "act"]
            self.sp_project = nn.Sequential(
                timm_layers.create_conv2d(out_channels, in_channels, 1, padding="same"),
                norm_layer_2d(in_channels),
                act_layer() if sp_project_method == "act" else nn.Identity(),
            )
            self.pixel_delta_ls = (
                LayerScale2d(in_channels, ls_init_value)
                if ls_init_value is not None
                else nn.Identity()
            )
        else:
            self.sp_project = nn.Identity()
            self.pixel_delta_ls = nn.Identity()

        if not return_updated_pixel_features or pixel_refine_method == "identity":
            self.pixel_refine = nn.Identity()
        elif pixel_refine_method.startswith("sep_conv_"):
            kernel_size = int(pixel_refine_method.split("_")[-1])
            print(f"Using separable conv with residual, kernel size {kernel_size}")
            self.pixel_refine = PixelRefineSepConv(
                in_channels,
                kernel_size,
                act_layer=act_layer,
                norm_layer_2d=norm_layer_2d,
                use_shortcut=True,
            )
        elif pixel_refine_method.startswith("no_residual_sep_conv_"):
            kernel_size = int(pixel_refine_method.split("_")[-1])
            print(f"Using separable conv without residual, kernel size {kernel_size}")
            self.pixel_refine = PixelRefineSepConv(
                in_channels,
                kernel_size,
                act_layer=act_layer,
                norm_layer_2d=norm_layer_2d,
                use_shortcut=False,
            )
        else:
            raise ValueError(f"Unknown pixel_refine_method: {pixel_refine_method}")
        if cross_only == False:
            if isinstance(drop_path_rate, Sequence):
                assert len(drop_path_rate) == depth

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, out_channels)) if use_cls_token else None
        )
        self.num_prefix_tokens = 1 if use_cls_token else 0
        num_patches = superpixel_shape[0] * superpixel_shape[1]
        embed_len = num_patches + 1 if use_cls_token else num_patches
        self.pos_embed = (
            nn.Parameter(torch.randn(1, embed_len, out_channels) * 0.02)
            if use_pos_embed
            else None
        )

        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=out_channels,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    # drop=drop_rate,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, Sequence)
                        else drop_path_rate
                    ),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    init_values=ls_init_value,
                )
                for i in range(depth)
            ]
        )
        self.vis_sp_feature = vis_sp_feature
        self.output_dir = output_dir
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            timm_layers.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward_patchify(
        self, x: torch.Tensor, sp_features_last: Optional[torch.Tensor], prev_info
    ):
        if isinstance(self.patch_embed, nn.AvgPool2d):
            # NOTE(meijieru): doesn't use sp_features_last
            sp_features = self.patch_embed(x)
            return None, sp_features, x
        if prev_info and self.similarity_embedding:
            patch_embed = self.patch_embed
            for key, val in prev_info.items():
                if "similarities_pixel" in key:
                    similarities = prepare_similarities(
                        patch_embed, val, scale_factor=1, merge_multihead_similarities=True
                    )
                    labels = superpixel_ops.compute_hard_association(similarities)
                    pos_emb = build_2d_sincos_position_embedding(patches_resolution = patch_embed.superpixel_shape[0],embed_dim = 4,device = x.device)
                    pos_emb = pos_emb[...,0].unsqueeze(-1)
                    b, h, w = labels.shape
                    pos_emb = pos_emb.expand(-1,-1,x.shape[1])

                    # Flatten labels tensor and expand dimensions to (4, 640 * 640, 1)
                    labels_flat = labels.view(b, -1).unsqueeze(-1)

                    # Use advanced indexing to gather from pos_emb
                    labels_embedding = pos_emb[:, labels_flat].squeeze(-2).squeeze(0)

                    # Reshape back to original shape (4, 640, 640, 4)
                    labels_embedding = rearrange(
                        labels_embedding,
                        'b (h w) c->b c h w ',
                        h = h,
                        w = w)
                    # import h5py
                    # with h5py.File('embeddingss.h5', 'w') as f:
                    #     # Create datasets for each embedding
                    #     f.create_dataset('labels_embedding', data=labels_embedding.detach().cpu().numpy())
                    #     f.create_dataset('labels', data=labels.detach().cpu().numpy()) 
                    
                    x = x + labels_embedding
            # similarity_embedding = get_similarity_embedding(similarities, patch_embed, scale_factor = 4)
        if sp_features_last is not None:
            sp_features_last = self.sp_transform_last(sp_features_last)
        info, pixel_features, sp_features = self.patch_embed(x, sp_features_last)

        self._tokenization_info = {
            key: val for key, val in info.items() if "similarities" in key
        }
        return info, sp_features, pixel_features

    @property
    def tokenization_info(self) -> Optional[MutableMapping[str, torch.Tensor]]:
        return getattr(self, "_tokenization_info", None)

    def forward_pixelify(
        self, x: torch.Tensor, sp_features_unflatten: torch.Tensor, info: Any
    ) -> torch.Tensor:
        assert x is not None
        if isinstance(self.patch_embed, nn.AvgPool2d):
            pixel_delta = F.interpolate(
                sp_features_unflatten,
                scale_factor=self.patch_embed.kernel_size,
                mode="bilinear",
            )
        else:
            similarity = st.get_final_similarity(
                info,
                self.patch_embed.num_blocks,
                pixel=self.use_pixel_similarities,
                merge=False,
            )
            if similarity is None:
                raise ValueError("Similarity is None")

            b, num_heads, _, h, w = similarity.shape
            if num_heads == 1:
                pixel_delta = superpixel_ops.update_pixel_features(
                    None, sp_features_unflatten, similarity.squeeze(1)
                )
            else:
                _, c, _, _ = sp_features_unflatten.shape
                pixel_delta = superpixel_ops.update_pixel_features(
                    None,
                    st.reshape_and_transpose_for_attention_operation(
                        sp_features_unflatten, num_heads
                    ).flatten(end_dim=1),
                    similarity.flatten(end_dim=1),
                )
                pixel_delta = pixel_delta.reshape(b, c, h, w)
        res = x + self.pixel_delta_ls(pixel_delta)
        res = self.pixel_refine(res)
        return res

    def forward_blocks_range(
        self, x: torch.Tensor, start: int, end: int, attn_dict_list:list = None,
    ) -> torch.Tensor:
        gt = None
        for i in range(start, end):
            x = self.blocks[i](x)
            if self.merge_pos and i in self.merge_pos:
                if self.vis_sp_feature:
                    x_ = x.clone()
                x,attn_dict_list , gt = self.mergelayer[self.merge_pos.index(i)](
                    x,self.hw_shape, attn_dict_list=attn_dict_list, prev_token=gt)
                if self.vis_sp_feature:
                    self.visualize_sp_feature(sp_before = x_, sp_after = x)
                if self.group_embedding:
                    attn_map = get_attn_maps_v2(self.hw_shape, attn_dict_list)[-1]
                    attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                    group_result = attn_map.argmax(dim=1)
                    pos_emb = build_2d_sincos_position_embedding(patches_resolution = int(math.sqrt(attn_map.shape[1])),
                                                                 embed_dim = x.shape[2],device = x.device,
                                                                 temperature=5000)
                    b, h, w = group_result.shape
                    group_result_flat = group_result.view(b, -1).unsqueeze(-1)
                    group_result_embedding = pos_emb[:, group_result_flat].squeeze(0).squeeze(-2)
                    # import h5py
                    # with h5py.File('embeddings_2.h5', 'w') as f:
                    #     # Create datasets for each embedding
                    #     f.create_dataset('group_result', data=group_result.detach().cpu().numpy())
                    #     f.create_dataset('group_result_embedding', data=rearrange(
                    #         group_result_embedding,
                    #         'b (h w) c-> b h w c',
                    #         h = h,
                    #         w = h).detach().cpu().numpy())
                    x = x + group_result_embedding
        return x,attn_dict_list

    def add_pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            if self.pos_embed is not None:
                x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
        return x

    def forward_unflatten_sp_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_token is not None:
            raise ValueError("Cannot unflatten sp_features when using cls_token")

        b, n, c = x.shape
        assert n == self.superpixel_shape[0] * self.superpixel_shape[1]
        x = x.transpose(1, 2).reshape(b, c, *self.superpixel_shape)
        return x

    def forward(
        self,
        x: torch.Tensor,
        sp_features_last: Optional[torch.Tensor],
        attn_dict_list:list = None,
        prev_info = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:

        info, sp_features, pixel_features_middle = self.forward_patchify(
            x, sp_features_last, prev_info
        )
        prev_info = self._tokenization_info       
        assert sp_features.dim() == 4
        self.hw_shape = tuple(sp_features.shape[2:4])
        
        sp_features = self.sp_lift(sp_features)
        sp_features = sp_features.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.cross_only == False:
            sp_features = self.add_pos_embed(sp_features)

            # sp_features = self.blocks(sp_features)

            # [0, seg_block_idx) are the blocks for segmentation
            sp_features_seg, attn_dict_list = self.forward_blocks_range(sp_features, 0, self.seg_block_idx,attn_dict_list)
            sp_features, attn_dict_list = self.forward_blocks_range(
                sp_features_seg, self.seg_block_idx, len(self.blocks),attn_dict_list
            )
        else:
            sp_features_seg = sp_features
            
        sp_features_unflatten = None
        updated_pixel_features = None
        if self.return_updated_pixel_features:
            sp_features_unflatten = self.forward_unflatten_sp_features(sp_features)
            sp_features_unflatten_projected = self.sp_project(sp_features_unflatten)
            updated_pixel_features = self.forward_pixelify(
                x if not self.use_middle_pixel_features else pixel_features_middle,
                sp_features_unflatten_projected,
                info,
            )
            if isinstance(
                self.patch_embed, st.SuperPixelTokenizationCrossAttentionAsymmetry
            ):
                pass
            else:
                sp_features_unflatten = sp_features_unflatten_projected

        if self.unflatten_sp_features:
            if sp_features_unflatten is None:
                sp_features_unflatten = self.forward_unflatten_sp_features(sp_features)
            sp_features = sp_features_unflatten

        return (
            updated_pixel_features,
            sp_features,
            sp_features_seg,
            attn_dict_list,
            prev_info,
        )
    def visualize_sp_feature(self, sp_before, sp_after):
        import h5py
        import os.path as osp
        h = w = int(math.sqrt(sp_before.shape[1]))
        sp_before = rearrange(
            sp_before,
            'b (h w) c -> b c h w',
            h = h, w=w
        ) 
        sp_after = rearrange(
            sp_after,
            'b (h w) c -> b c h w',
            h = h, w=w
        )         
        out_file = osp.join(self.output_dir, 'vis_sp', 'sp.h5')
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        base_name, file_ext = osp.splitext(out_file)

        with h5py.File(out_file,'a') as f:
            keys = list(f.keys())
            key = "sp_before"
            original_key = key
            count = int(0)
            while key in keys:
                print(f"Dataset with key {key} already exists. Updating key name.")
                count += 1
                key = original_key + str(count)
            f.create_dataset(key,data=sp_before.detach().cpu().numpy())                 
            key = "sp_after"
            original_key = key
            count = int(0)
            while key in keys:
                print(f"Dataset with key {key} already exists. Updating key name.")
                count += 1
                key = original_key + str(count)
            f.create_dataset(key,data=sp_after.detach().cpu().numpy()) 
        
            




class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

@HEADS.register_module()
class SuperformerBottleNeck(BaseDecodeHead):
    def __init__(
        self,
        img_size: Sequence[int] = (224,224),
        in_channels: int = 3,
        num_classes: int = 1000,
        stem_channels_list: Sequence[int] = (64,),
        stem_kernel_sizes: Sequence[int] = (4,),
        stem_strides: Sequence[int] = (4,),
        stem_conv_types: Sequence[str] = ("patch",),
        depths: Sequence[int] = (2, 10),
        dims: Sequence[int] = (384, 384),
        heads: Sequence[int] = (6, 6),
        strides: Sequence[int] = (1, 1),
        sp_sizes: Sequence[int] = (2, 2),
        sp_heads: Sequence[int] = (1, 1),
        sp_method: str = "sp_cross",
        sp_iter: int = 2,
        sp_features_init_methods: Sequence[str] = ("avgpool", "from_feature"),
        sp_global_init_method: str = "none",
        sp_position_embedding_method: str = "depthwise",
        sp_position_embedding_stride: int = -1,
        sp_sp_position_embedding_stride: Optional[int] = None,
        sp_pixel_features_update_method: str = "residual",
        sp_kwargs: Optional[Dict[str, Any]] = None,
        conv_norm_layer=None,
        conv_act_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        ls_init_value: Optional[float] = 1e-5,
        sp_ls_init_value: Optional[float] = 1e-5,
        pre_norm_pixel: bool = False,
        pixel_refine_method: str = "identity",
        sp_embed_method: str = "identity",
        sp_project_method: str = "noact",
        class_token_position_method: str = "none",
        pos_embed_position_method: str = "every_stage",
        no_embed_class: bool = False,
        weight_init: str = "default",
        use_pixel_similarities: bool = False,
        use_middle_pixel_features: bool = False,
        seg_specific_classifier: str = None,
        seg_num_classes: int = 150,
        use_stem: bool = True,
        merge_heads: Sequence[int] = (2,),
        num_group_tokens: Sequence[int] = (8,),
        num_output_groups: Sequence[int] = (8,),
        hard_assignment: Sequence[bool] = (True,),
        drop_rate_merge: float = 0.,
        attn_drop_rate_merge: float = 0.,
        drop_path_rate_merge: float = 0.1,
        depths_merge: Sequence[int] = (6,),
        token_strides: Sequence[int] = None,
        cross_only: Sequence[bool] = (False,False,True,),
        token_channels_list: Sequence[int] = (2,),
        vis_token: str = None,
        vis_featuremap: bool = False,
        vis_sp_logits: bool = False,
        output_dir: str = None,
        group_token_init_methods: Sequence[str] = ("avgpool", "learnable",),
        group_token_embedding_method: str = "learnable",
        group_projector_from_sp_feature: bool=False,
        merge_methods:str = "vanilla",
        merge_pos: Sequence[int] = None,
        merge_updated_similarity: bool = False,
        group_projector_methonds:str = 'linear',
        classification_feature :bool = 'superpixel',
        arch_settings: dict = {
            'embed_dims': 216,
            'patch_size': 8,
            'window_size': 2,
            'num_layers': 4,
            'num_heads': 12,
            'num_group_heads': 6,
            'num_group_forward_heads': 6,
            'num_ungroup_heads': 6,
            'ffn_ratio': 4.,
            'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),
            'mlpmixer_depth': 1,
            'group_layers': {0:64,1:32,2:32,3:16},
            'drop_path_rate': 0.2
        },
        similarity_embedding = False,
        group_embedding = False,
        vis_sp_feature: bool = False,
        **kwargs

    ):
        super().__init__(in_channels=3,
        channels=256,
        num_classes=150,
        out_channels=150,
        in_index=0,
**kwargs)

        if sp_pixel_features_update_method == "fixed" and use_pixel_similarities:
            raise ValueError("pixel_similarities won't be updated")
        self.img_size = img_size
        self.sp_method = sp_method
        self.sp_iter = sp_iter
        self.sp_dim=dims[0]
        self.sp_features_init_methods = sp_features_init_methods
        self.sp_position_embedding_method = sp_position_embedding_method
        self.sp_position_embedding_stride = sp_position_embedding_stride
        self.sp_sp_position_embedding_stride = sp_sp_position_embedding_stride
        self.sp_pixel_features_update_method = sp_pixel_features_update_method
        self.sp_ls_init_value = sp_ls_init_value
        self.sp_kwargs = sp_kwargs or {}
        self.use_middle_pixel_features = use_middle_pixel_features

        self.class_token_position_method = class_token_position_method
        self.no_embed_class = no_embed_class
        self.pos_embed_position_method = pos_embed_position_method

        act_layer = act_layer or nn.GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # norm_layer_2d = partial(get_norm_layer_2d(norm_layer), eps=1e-6)
        norm_layer_2d = partial(get_norm_layer_2d(norm_layer))

        self.norm_layer = norm_layer
        self.norm_layer_2d = norm_layer_2d
        
        self.seg_num_classes = seg_num_classes
        self.seg_specific_classifier = seg_specific_classifier

        conv_norm_layer = conv_norm_layer or partial(timm_layers.LayerNorm2d, eps=1e-6)
        conv_act_layer = conv_act_layer or act_layer
        
        stem_kernel_sizes = stem_kernel_sizes or [3] * len(stem_channels_list)
        self.use_stem=use_stem
        if self.use_stem is True:
            self.stem = st.ConvStem(
                in_channels,
                stem_channels_list,
                stem_kernel_sizes,
                stem_strides,
                conv_types=stem_conv_types,
                # norm_layer=conv_norm_layer,  # pyright: ignore [reportGeneralTypeIssues]
                # act_layer=conv_act_layer,  # pyright: ignore [reportGeneralTypeIssues]
            )

        cur_stride = int(np.prod(stem_strides))
        # pixel_output_shape = tuple([img_size // cur_stride for _ in range(2)])
        
        pixel_dim=stem_channels_list[-1]
        sp_dim = None
        self.sp_global_init_method = sp_global_init_method
        if self.sp_features_init_methods[0] == "from_feature":
            assert self.sp_method in ["sp_cross_asymmetry"]
            sp_dim = dims[0]
            if self.sp_global_init_method == "patch":
                self.sp_init = nn.Sequential(
                    timm_layers.create_conv2d(
                        in_channels=pixel_dim,
                        out_channels=sp_dim,
                        kernel_size=sp_sizes[0],
                        stride=sp_sizes[0],
                        padding="same",
                        bias=False,
                    ),
                    norm_layer_2d(sp_dim),
                )
            elif self.sp_global_init_method == "avgpool_lift":
                self.sp_init = nn.Sequential(
                    nn.AvgPool2d(kernel_size=sp_sizes[0], stride=sp_sizes[0]),
                    timm_layers.create_conv2d(pixel_dim, sp_dim, 1, padding="same"),
                    norm_layer_2d(sp_dim),
                    act_layer(),
                )
            elif self.sp_global_init_method == "lift_avgpool":
                self.sp_init = nn.Sequential(
                    timm_layers.create_conv2d(pixel_dim, sp_dim, 1, padding="same"),
                    norm_layer_2d(sp_dim),
                    act_layer(),
                    nn.AvgPool2d(kernel_size=sp_sizes[0], stride=sp_sizes[0]),
                )
            elif self.sp_global_init_method in ["conv"]:
                self.sp_init = nn.Sequential(
                    timm_layers.create_conv2d(
                        in_channels=pixel_dim,
                        out_channels=sp_dim,
                        kernel_size=sp_sizes[0],
                        stride=sp_sizes[0],
                        padding="same",
                        bias=False,
                    ),
                    norm_layer_2d(sp_dim),
                    act_layer(),
                )
            else:
                raise ValueError(
                    f"Unknown sp_global_init_method: {sp_global_init_method}"
                )
        else:
            assert self.sp_global_init_method == "none"

        assert (
            len(depths)
            == len(dims)
            == len(heads)
            == len(sp_sizes)
            == len(sp_heads)
            == len(sp_features_init_methods)
        )

        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate_merge, sum(depths_merge))]

        cur_depth = 0

        if self.pos_embed_position_method == "every_stage":
            use_pos_embeds = [True] * len(depths)
        elif self.pos_embed_position_method == "none":
            use_pos_embeds = [False] * len(depths)
        else:
            raise ValueError(
                f"Unknown pos_embed_position_method: {self.pos_embed_position_method}"
            )

        if self.class_token_position_method == "last_stage":
            use_class_tokens = [False] * (len(depths) - 1) + [True]
        elif self.class_token_position_method == "none":
            use_class_tokens = [False] * len(depths)
        else:
            raise ValueError(
                f"Unknown class_token_position_method: {self.class_token_position_method}"
            )
        
        self.merge_methods = merge_methods
        mergelayers = None
        if self.merge_methods =="GPViT_mix":
            self.arch_settings = arch_settings
            self.merge_pos = merge_pos
        elif self.merge_methods in ["GPViT_post","GPViT_post_parallel"]:
            self.arch_settings = arch_settings
            self.merge_pos = None         
        self.num_layers = self.arch_settings['num_layers']
        
        mergelayers = nn.ModuleList()
        import copy
        _arch_settings = copy.deepcopy(self.arch_settings)


        num_stages = len(depths)
        stages = []


        stage_in_dim = stem_channels_list[-1]
        assert classification_feature in {'superpixel','pixel','combined'}
        self.classification_feature = classification_feature
        dpr = [x.item() for x in torch.linspace(0, self.arch_settings['drop_path_rate'], self.arch_settings['num_layers'])]
        if self.merge_methods in ["GPViT_post",'GPViT_post_parallel']:
                self.mergelayer = self._make_merge_layer(
                None, _arch_settings
                )
        if self.merge_methods == 'GPViT_post_parallel':
            self.bottleneck = nn.Sequential(
                nn.Conv2d(self.num_layers * dims[-1], 
                dims[-1],
                 kernel_size = 3,
                padding =1,),
                timm_layers.LayerNorm2d(dims[-1]),
                nn.GELU(),
            )
        if self.classification_feature in ['pixel','combined']:
            self.use_pixel_classification = True
        else:
            self.use_pixel_classification = False

        for i, (depth, dim, head, sp_size, sp_head, stride) in enumerate(
            zip(depths, dims, heads, sp_sizes, sp_heads, strides)
        ):
            cur_stride *= stride
            sp_stride = sp_size * stride // sp_sizes[i - 1] if i > 0 else 1
            sp_layer, sp_shape = self._make_superpixel_layer(
                i, cur_stride, sp_size, sp_head, pixel_dim, dim, sp_method
            )
            if self.merge_methods == "GPViT_mix":
                mergelayer = self._make_merge_layer(
                    i, _arch_settings
                )
            # first feature already has norm & act
            pre_norm_pixel_stage = pre_norm_pixel and i != 0
            if pre_norm_pixel_stage:
                print(f"pre_norm_pixel for stage {i}")
            if self.use_pixel_classification is False:
                return_updated_pixel_features = i < num_stages - 1
            else:
                return_updated_pixel_features = i < num_stages
            unflatten_sp_features = return_updated_pixel_features

            print(
                f"stage {i}: use_cls_token: {use_class_tokens[i]}, use_pos_embed: {use_pos_embeds[i]}"
            )

            stages.append(
                SuperformerStage(
                    stage_in_dim,
                    sp_dim,
                    dim,
                    stride,
                    sp_stride,
                    head,
                    depth,
                    norm_layer=norm_layer,
                    norm_layer_2d=norm_layer_2d,
                    act_layer=act_layer,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rates[cur_depth : cur_depth + depth],
                    attn_drop_rate=attn_drop_rate,
                    superpixel_layer=sp_layer,
                    ls_init_value=ls_init_value,
                    pre_norm_pixel=pre_norm_pixel_stage,
                    sp_embed_method=sp_embed_method,
                    sp_project_method=sp_project_method,
                    pixel_refine_method=pixel_refine_method,
                    return_updated_pixel_features=return_updated_pixel_features,
                    unflatten_sp_features=unflatten_sp_features,
                    use_pos_embed=use_pos_embeds[i],
                    use_cls_token=use_class_tokens[i],
                    no_embed_class=self.no_embed_class,
                    superpixel_shape=sp_shape,
                    use_pixel_similarities=use_pixel_similarities,
                    use_middle_pixel_features=use_middle_pixel_features,
                    cross_only=cross_only[i],
                    mergelayer=merge_layer if self.merge_methods == "GPViT_mix" else None,
                    merge_pos = merge_pos[i] if self.merge_methods == "GPViT_mix" else None,
                    similarity_embedding = similarity_embedding,
                    group_embedding= group_embedding,
                    vis_sp_feature=vis_sp_feature,
                    output_dir = output_dir,
                )
            )
            cur_depth += depth

        assert cur_depth == sum(depths)
        self.stages = nn.ModuleList(stages)

        # Classifier Head
        self.embed_dim = dims[-1]

        has_class_token = class_token_position_method != "none"
        use_fc_norm = not has_class_token
        self.num_prefix_tokens = 1 if has_class_token else 0
        self.norm = norm_layer(self.embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(self.embed_dim) if use_fc_norm else nn.Identity()
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if self.seg_specific_classifier:
            assert seg_num_classes > 0
            # NOTE(meijieru): the token features doesn't use avgpool, so we skip
            # the fc_norm.
            if self.classification_feature =='superpixel':
                if self.seg_specific_classifier == "Linear":
                    if token_strides is None:
                        self.seg_head = nn.Linear(self.embed_dim, self.seg_num_classes)
                    else:
                        self.seg_head = nn.Linear(token_channels_list[-1], self.seg_num_classes)
                elif self.seg_specific_classifier == "Conv":
                    if token_strides is None:
                        self.seg_head = nn.Conv2d(self.embed_dim, self.seg_num_classes,1)
                    else:
                        self.seg_head = nn.Conv2d(token_channels_list[-1], self.seg_num_classes,1)
                else:
                    print(self.seg_specific_classifier)
                    raise ValueError()
                if token_strides is None:
                    self.seg_norm = norm_layer(self.embed_dim)
                else:
                    self.seg_norm = norm_layer(token_channels_list[-1])
            elif self.classification_feature =='pixel':
                self.seg_norm = norm_layer(stem_channels_list[-1])
                self.seg_head = nn.Conv2d(stem_channels_list[-1], self.seg_num_classes,1)
            elif self.classification_feature =='combined':
                from mmcv.cnn import ConvModule

                self.conv_cat = ConvModule(
                self.embed_dim + stem_channels_list[-1],
                self.embed_dim + stem_channels_list[-1],
                kernel_size=3,
                padding=3 // 2,
                )
                self.seg_norm = norm_layer( stem_channels_list[-1] + self.embed_dim)
                self.seg_head = nn.Conv2d(stem_channels_list[-1] + self.embed_dim, self.seg_num_classes,1)

        if weight_init != "skip":
            self.init_weights(weight_init)
        
        
        #downsample module from groupvit
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate_merge, sum(depths_merge))]
        self.vis_token = vis_token
        self.output_dir = output_dir
        self.vis_featuremap = vis_featuremap
        self.vis_sp_logits = vis_sp_logits
        self.vis_sp_feature_v2 = vis_sp_feature
    def init_weights_mergelayers(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.zeros_(m.weight)
  
    def init_weights(self, mode=""):
        assert mode in (
            "jax",
            "jax_nlhb",
            "moco",
            "",
            "default",
        )
        if mode == "default" or mode == "":
            init_fn = init_superformer_weights
        else:
            raise NotImplementedError(f"Unknown init mode: {mode}")
        
        timm.models.named_apply(init_fn, self)
        
        # timm.models.helpers.named_apply(init_fn, self)

    def _make_superpixel_layer(
        self,
        i: int,
        pixel_stride: int,
        sp_size: int,
        sp_head: int,
        pixel_dim: int,
        sp_dim: Optional[int],
        method: str = "sp_cross",
    ) -> Tuple[Callable, Sequence[int]]:
        assert self.img_size[0] % pixel_stride == 0
        pixel_shape = [self.img_size[i] // pixel_stride for i in range(2)]
        sp_shape = [val // sp_size for val in pixel_shape]

        if (
            i != len(self.sp_features_init_methods) - 1
            and self.use_middle_pixel_features 
            or self.use_pixel_classification
        ):
            return_final_pixel_features = True
        else:
            return_final_pixel_features = False
        if method == "sp_cross":
            sp_fn = partial(
                st.SuperPixelTokenizationCrossAttention,
                self.sp_iter,
                sp_head,
                pixel_dim,
                pixel_shape,
                sp_shape,
                superpixel_features_init_method=self.sp_features_init_methods[i],
                position_embedding_method=self.sp_position_embedding_method,
                position_embedding_stride=self.sp_position_embedding_stride,
                sp_position_embedding_stride=self.sp_sp_position_embedding_stride,
                pixel_features_update_method=self.sp_pixel_features_update_method,
                return_final_pixel_features=return_final_pixel_features,
                layer_kwargs={
                    "ls_init_value": self.sp_ls_init_value,
                    "norm_layer": self.norm_layer_2d,
                },
                **self.sp_kwargs,
            )
        elif method == "sp_cross_asymmetry":
            assert sp_dim is not None
            sp_fn = partial(
                st.SuperPixelTokenizationCrossAttentionAsymmetry,
                self.sp_iter,
                sp_head,
                pixel_dim,
                pixel_shape,
                sp_shape,
                superpixel_features_init_method=self.sp_features_init_methods[i],
                position_embedding_method=self.sp_position_embedding_method,
                position_embedding_stride=self.sp_position_embedding_stride,
                sp_position_embedding_stride=self.sp_sp_position_embedding_stride,
                pixel_features_update_method=self.sp_pixel_features_update_method,
                return_final_pixel_features=return_final_pixel_features,
                num_channels_sp=sp_dim,
                num_heads_sp=sp_head,
                layer_kwargs={
                    "ls_init_value": self.sp_ls_init_value,
                    "norm_layer": self.norm_layer_2d,
                },
                **self.sp_kwargs,
            )
        elif method == "patch":
            sp_fn = partial(nn.AvgPool2d, kernel_size=sp_size, stride=sp_size)
        else:
            raise ValueError(f"Unknown superpixel method {method}")
        return sp_fn, sp_shape
    def _make_merge_layer(
        self,
        i,
        _arch_settings:dict,
        
    ):
        if self.merge_methods == "GPViT_mix":
            depth = len(self.merge_pos[i])
            mergelayer = nn.ModuleList()
            for j in range(depth):
                if _arch_settings["group_projector_methonds"] == 'linear':
                    group_projector =nn.Sequential(
                        nn.LayerNorm(_arch_settings['embed_dims']),
                        MixerMlp(_arch_settings['group_layers'][i-1], _arch_settings['embed_dims'] // 2, _arch_settings['group_layers'][i])
                        )
                elif _arch_settings["group_projector_methonds"] == 'cross':
                    group_projector = FullAttnCatBlock(
                        embed_dims=_arch_settings['embed_dims'],
                        num_heads = _arch_settings['num_group_heads'],
                        key_is_query=False,
                        value_is_key=False,
                    )
                elif _arch_settings["group_projector_methonds"]== None:
                    group_projector=None
                else :
                    raise(NotImplementedError)

            
                _layer_cfg = dict(
                        embed_dims=_arch_settings['embed_dims'],
                        depth=_arch_settings['mlpmixer_depth'],
                        num_group_heads=_arch_settings['num_group_heads'],
                        num_forward_heads=_arch_settings['num_group_forward_heads'],
                        num_ungroup_heads=_arch_settings['num_ungroup_heads'],
                        num_group_token=_arch_settings['group_layers'][j],
                        ffn_ratio=_arch_settings['ffn_ratio'],
                        with_cp=None,
                        group_projector=group_projector,
                        zero_init_group_token=group_projector is not None,
                        group_projector_methonds = _arch_settings["group_projector_methonds"],
                        association_embedding = _arch_settings["association_embedding"],
                        group_token_init_method = _arch_settings["group_token_init_method"])
                group_layer = GPBlock(**_layer_cfg)
                mergelayer.append(group_layer)
        elif self.merge_methods in ["GPViT_post","GPViT_post_parallel"]:
            mergelayer = nn.ModuleList()
            for i in range(0, _arch_settings['num_layers']):
                if i > 0:
                    if _arch_settings["group_projector_methonds"] == 'linear':
                        group_projector =nn.Sequential(
                            nn.LayerNorm(_arch_settings['embed_dims'][i]),
                            MixerMlp(_arch_settings['group_layers'][i-1], _arch_settings['embed_dims'][i-1] // 2, _arch_settings['group_layers'][i])
                            )
                    elif _arch_settings["group_projector_methonds"] == 'cross':
                        group_projector = FullAttnCatBlock(
                            embed_dims=_arch_settings['embed_dims'],
                            num_heads = _arch_settings['num_group_heads'],
                            key_is_query=False,
                            value_is_key=False,
                        )
                    elif _arch_settings["group_projector_methonds"]== None:
                        group_projector=None
                    else :
                        raise(NotImplementedError)
                else:
                    group_projector=None
            
                _layer_cfg = dict(
                        embed_dims=_arch_settings['embed_dims'][i],
                        depth=_arch_settings['mlpmixer_depth'],
                        num_group_heads=_arch_settings['num_group_heads'][i],
                        num_forward_heads=_arch_settings['num_group_forward_heads'][i],
                        num_ungroup_heads=_arch_settings['num_ungroup_heads'][i],
                        num_group_token=_arch_settings['group_layers'][i],
                        ffn_ratio=_arch_settings['ffn_ratio'],
                        with_cp=None,
                        group_projector=group_projector,
                        zero_init_group_token=True,
                        group_projector_methonds = _arch_settings["group_projector_methonds"],
                        association_embedding = _arch_settings["association_embedding"],
                        group_token_init_method = _arch_settings["group_token_init_method"][i],
                        init_kernel_size = _arch_settings["init_kernel_sizes"][i],
                        init_stride = _arch_settings["init_strides"][i],
                        use_assign =  _arch_settings["use_assign"])
                group_layer = GPBlock(**_layer_cfg)
                mergelayer.append(group_layer)
                
            
        return mergelayer
    
    def init_superpixel_features(
        self, pixel_features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.sp_features_init_methods[0] == "from_feature":
            return self.sp_init(pixel_features)
        elif self.sp_features_init_methods[0] == "leanable":
            return nn.Parameter(
                torch.randn(1, self.sp_dim, self.superpixel_shape[0],self.superpixel_shape[1]) * 0.02
            )


        return None
    def init_grouptoken(
        self, pixel_features: torch.Tensor, num_group_tokens: Sequence[int],dim: int
    ) -> Optional[torch.Tensor]:
        token_size=math.sqrt(num_group_tokens[0])
        _, pixel_dim, _, _ = pixel_features.shape
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        norm_layer_2d = partial(get_norm_layer_2d(norm_layer))


        self.token_init = nn.Sequential(
                timm_layers.create_conv2d(pixel_dim, dim, 1, padding="same"),
                norm_layer_2d(dim),
                nn.GELU(),
                nn.AvgPool2d(kernel_size=token_size, stride=token_size),
            )

        return self.token_init(pixel_features)

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, MutableMapping[str, torch.Tensor]]:
        if self.use_stem == True:
            endpoints, pixel_features = self.stem(x)
        else:
            endpoints = None
            pixel_features = x
        sp_features_last = self.init_superpixel_features(pixel_features)
        assert len(self.stages) > 0
        if self.merge_methods in ["GPViT_mix","GPViT_post","GPViT_post_parallel"]:
            attn_dict_list  = []
            prev_info = None
            for i, stage in enumerate(self.stages):
                        pixel_features, sp_features, sp_features_seg, attn_dict_list, prev_info = stage(
                            pixel_features,
                            sp_features_last,
                            attn_dict_list,
                            prev_info,
                        )                

                        sp_features_last = sp_features
            if self.classification_feature in {'pixel','combined'}:            
                return sp_features, sp_features_seg,endpoints,attn_dict_list,pixel_features
            else:
                return sp_features, sp_features_seg,endpoints,attn_dict_list
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        assert x.dim() == 3
        if self.class_token_position_method == "none":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        else:
            x = x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward_segmentation(
        self, x: torch.Tensor, pixel_feature: torch.Tensor = None, attn_dict_list = None, return_pixel_logits: bool = True, stride: int = 2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.classification_feature == 'superpixel':
            if self.merge_methods in ["GPViT_mix","GPViT_post","GPViT_post_parallel"]:
                x = self.seg_norm(x)
                if not self.seg_specific_classifier:
                    raise ValueError("No segmentation head is found.")
                elif self.seg_specific_classifier=="Linear":
                    b, num, c = x.shape

                    x = x.reshape(b * num, c)

                    sp_logits = self.seg_head(x)
                    sp_logits = sp_logits.reshape(b, num, -1)
                    
                last_stage = self.stages[-1]
                last_sp_layer = last_stage.patch_embed

                if isinstance(last_sp_layer, nn.AvgPool2d):
                    sh=sw = last_sp_layer.kernel_size
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    sh, sw = last_sp_layer.superpixel_shape
                else:
                    raise ValueError()
                sp_logits = sp_logits.view(b, sh, sw, -1).permute(0, 3, 1, 2)

                pixel_logits = None
                if return_pixel_logits:
                    if isinstance(last_sp_layer, nn.AvgPool2d):
                        raise NotImplementedError()
                    elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                        info = last_stage.tokenization_info
                        if info is None:
                            raise ValueError()
                        scale_factor = self.img_size[0] // last_sp_layer.pixel_shape[0] // stride

                        # raise NotImplementedError(
                        #     "TODO(meijier): use pixel similarities & not merge"
                        # )

                        similarities = st.get_final_similarity(info, last_sp_layer.num_blocks)
                        similarities = prepare_similarities(
                            last_sp_layer,
                            similarities,  # pyright: ignore [reportGeneralTypeIssues]
                            scale_factor=scale_factor,
                        )
                        similarities = similarities.softmax(1)
                        similarities = einops.rearrange(
                            similarities, "b n sh ph sw pw -> b n (sh ph) (sw pw)"
                        )
                        pixel_logits = superpixel_ops.expand_superpixel_features(
                            sp_logits, similarities
                        )
                    else:
                        raise ValueError()

                return sp_logits, pixel_logits

        elif self.classification_feature == 'pixel':
            pixel_feature = self.seg_norm(rearrange(pixel_feature,
                                        "b c h w -> b h w c"))
            pixel_logits = self.seg_head(rearrange(pixel_feature,
                                        "b h w c -> b c h w  "))
            return None, pixel_logits
        else:
            _, n, _ = x.shape
            x = einops.rearrange(
                x,
                'b (h w) c-> b c h w',
                h = int(math.sqrt(n)),
                w = int(math.sqrt(n)),
            )
            
            x = F.interpolate(x,size=pixel_feature.size()[-2:],mode='bilinear' )
            x = self.conv_cat(torch.cat([x, pixel_feature], dim=1))

            x = self.seg_norm(
                einops.rearrange(
                    x,
                    'b c h w ->b h w c',
                ))
            x = einops.rearrange(
                x,
                'b h w c -> b c h w',
            )     
            x = self.seg_head(x)
            return None, x

    def forward(
        self,
        x: torch.Tensor,
        generate_seg: bool = True,
        return_pixel_logits: bool = True,
        seg_stride: int = 1,

    ) -> Union[torch.Tensor, MutableMapping[str, torch.Tensor]]:

        # use group token module from GPViT to enhance
        if self.merge_methods =="GPViT_mix":
            # the type of feature used for classification
            # attn_dict_list is to record association between group token and superpixel
            if self.classification_feature in {'pixel','combined'}:            
                sp_features, sp_features_seg, endpoints,attn_dict_list, pixel_feature = self.forward_features(x)
            else:
                sp_features, sp_features_seg, endpoints,attn_dict_list = self.forward_features(x)
                pixel_feature = None
            if self.vis_token != None:
                if sp_features.dim() == 3:
                    B, N, C = sp_features.shape
                    pw = ph = int(math.sqrt(N))
                    sp_shape = (B, C, ph, pw)
                else:
                    sp_shape = sp_features.shape
                img = x.detach()
                _, _, ih, iw = img.shape

                self.std = [58.395, 57.12, 57.375]
                self.mean = [123.675, 116.28, 103.53]
                img = img* torch.tensor(self.std).reshape(1, 3, 1, 1).cuda() + torch.tensor(self.mean).reshape(1, 3, 1, 1).cuda()
                img = img.cpu().numpy()
                self.visualize_grouptoken_v2(img, sp_shape, attn_dict_list, self.output_dir)
                
                self.visualize_superpixel(img, resize_similarities= True, output_dir= self.output_dir, attn_dict_list = attn_dict_list, sp_shape = sp_shape)

            if generate_seg:
                
                sp_logits, pixel_logits = self.forward_segmentation(
                x = sp_features_seg,
                pixel_feature = pixel_feature,
                return_pixel_logits = return_pixel_logits, 
                stride = seg_stride,
            )
                ret={}
                ret["seg"] = pixel_logits
                return ret["seg"]
        elif  self.merge_methods =="GPViT_post":
            sp_features, sp_features_seg, endpoints,attn_dict_list = self.forward_features(x)
            pixel_feature = None

            gt = None
            for i, layer in enumerate(self.mergelayer):
                sp_features_seg ,attn_dict_list , gt = layer(
                    sp_features_seg,self.stages[-1].patch_embed.superpixel_shape, attn_dict_list=attn_dict_list, prev_token=gt)
                if self.vis_sp_feature_v2:
                    self.visualize_sp_feature_v2(sp_features_seg,gt)                
                
            if self.vis_token != None:
                if sp_features.dim() == 3:
                    B, N, C = sp_features.shape
                    pw = ph = int(math.sqrt(N))
                    sp_shape = (B, C, ph, pw)
                else:
                    sp_shape = sp_features.shape
                img = x.detach()
                _, _, ih, iw = img.shape

                self.std = [58.395, 57.12, 57.375]
                self.mean = [123.675, 116.28, 103.53]
                img = img* torch.tensor(self.std).reshape(1, 3, 1, 1).cuda() + torch.tensor(self.mean).reshape(1, 3, 1, 1).cuda()
                img = img.cpu().numpy()
                self.visualize_grouptoken_v2(img, sp_shape, attn_dict_list, self.output_dir)
                
                self.visualize_superpixel(img, resize_similarities= True, output_dir= self.output_dir, attn_dict_list = attn_dict_list, sp_shape = sp_shape)
            if generate_seg:
                
                sp_logits, pixel_logits = self.forward_segmentation(
                x = sp_features_seg,
                pixel_feature = pixel_feature,
                return_pixel_logits = return_pixel_logits, 
                stride = seg_stride,
            )
                ret={}
                ret["seg"] = pixel_logits
                return ret["seg"]
        elif  self.merge_methods =="GPViT_post_parallel":
            sp_features, sp_features_seg, endpoints,attn_dict_list = self.forward_features(x)
            pixel_feature = None
            sp_feature_list = []
            for i, layer in enumerate(self.mergelayer):
                if self.vis_sp_feature_v2:
                    self.visualize_sp_feature_v2(sp_features_seg,gt)    
                gt = None
                sp_features_seg ,attn_dict_list , gt = layer(
                    sp_features_seg,self.stages[-1].patch_embed.superpixel_shape, attn_dict_list=attn_dict_list, prev_token=gt)
                
                sp_feature_list.append(rearrange(
                    sp_features_seg,
                    'b (h w) c -> b c h w',
                    h = self.stages[-1].patch_embed.superpixel_shape[-1],
                    w = self.stages[-1].patch_embed.superpixel_shape[-2]))
                
            sp_features_seg = self.bottleneck(torch.cat(sp_feature_list,dim = 1))
            sp_features_seg = rearrange(sp_features_seg,
                    'b c h w -> b (h w) c')
            if self.vis_token != None:
                if sp_features.dim() == 3:
                    B, N, C = sp_features.shape
                    pw = ph = int(math.sqrt(N))
                    sp_shape = (B, C, ph, pw)
                else:
                    sp_shape = sp_features.shape
                img = x.detach()
                _, _, ih, iw = img.shape

                self.std = [58.395, 57.12, 57.375]
                self.mean = [123.675, 116.28, 103.53]
                img = img* torch.tensor(self.std).reshape(1, 3, 1, 1).cuda() + torch.tensor(self.mean).reshape(1, 3, 1, 1).cuda()
                img = img.cpu().numpy()
                self.visualize_grouptoken_v2(img, sp_shape, attn_dict_list, self.output_dir)
                
                self.visualize_superpixel(img, resize_similarities= True, output_dir= self.output_dir, attn_dict_list = attn_dict_list, sp_shape = sp_shape)
            if generate_seg:
                
                sp_logits, pixel_logits = self.forward_segmentation(
                x = sp_features_seg,
                pixel_feature = pixel_feature,
                return_pixel_logits = return_pixel_logits, 
                stride = seg_stride,
            )
                ret={}
                ret["seg"] = pixel_logits
                return ret["seg"]
        else:          
            sp_features, sp_features_seg, grouped_img_tokens, attn_dict_list, endpoints = self.forward_features(x)
        
            if self.vis_token != None:
                B, N, C = sp_features.shape
                pw = ph = int(math.sqrt(N))
                sp_shape = (B, C, ph, pw)
                
                img = x.detach()
                _, _, ih, iw = img.shape

                self.std = [58.395, 57.12, 57.375]
                self.mean = [123.675, 116.28, 103.53]
                img = img* torch.tensor(self.std).reshape(1, 3, 1, 1).cuda() + torch.tensor(self.mean).reshape(1, 3, 1, 1).cuda()
                img = img.cpu().numpy()
                self.visualize_grouptoken_v2(img, sp_shape, attn_dict_list, self.output_dir)
                
                self.visualize_superpixel(img, resize_similarities= True, output_dir= self.output_dir, attn_dict_list = attn_dict_list, sp_shape = sp_shape)
            if generate_seg:
                    sp_logits, pixel_logits = self.forward_segmentation(
                    sp_features_seg, grouped_img_tokens, attn_dict_list, return_pixel_logits, stride=seg_stride
                )
                    ret={}
                    ret["seg"] = pixel_logits
                    return ret["seg"]
            else:
                logits = self.forward_head(sp_features)
                return logits

    def no_weight_decay(self):
        no_weight_decay = set()
        for name, _ in self.named_parameters():
            if (
                "pos_embed" in name
                or "init_superpixel_queries" in name
                or "cls_token" in name
            ):
                no_weight_decay.add(name)
        print(f"no_weight_decay: {no_weight_decay}")
        return no_weight_decay
    def visualize_superpixel(self, img, resize_similarities: bool = True, output_dir: str = None,attn_dict_list = None, sp_shape = None):
        import os.path as osp
        from PIL import Image
        out_file = osp.join(output_dir, 'vis_tokens', self.vis_token, f'{self.vis_token}.jpg')
        
        _, _, ih, iw = img.shape
        im_image = Image.fromarray(img.squeeze().transpose(1, 2, 0).astype(np.uint8))
        res = {}
        for i, stage in enumerate(self.stages):
            patch_embed = stage.patch_embed
            tokenization_info = stage.tokenization_info
            if resize_similarities:
                scale_factor = self.img_size[0] // stage.patch_embed.pixel_shape[0]
            else:
                scale_factor = 1
            if tokenization_info is None:
                continue
            res_i = self.get_superpixel(tokenization_info, patch_embed, scale_factor, attn_dict_list, sp_shape)
            res.update({f"stage{i}_{key}": val for key, val in res_i.items()})
        

        counter = 0
        base_name, file_ext = osp.splitext(out_file)
        for key, val in res.items():
            
            im = val.numpy().astype(np.uint8)
            resize_output = not resize_similarities or im.shape[0] != ih
            im = Image.fromarray(im)
            if resize_output:
                print('resize_output')
                im = im.resize((ih, iw), Image.Resampling.NEAREST)
                
            alpha = 0.3
            im_alpha = np.array(im_image, dtype=np.float32) * alpha + np.array(im, dtype=np.float32) * (1 - alpha)
            im_alpha = Image.fromarray(im_alpha.astype(np.uint8))
            while osp.exists(f"{base_name}_{counter}_vis_{i}_{key}_{file_ext}"):
                counter += 1
            img_out_file = f"{base_name}_{counter}_vis_{i}_{key}_{file_ext}"
            directory, _ = os.path.split(img_out_file)
            if not os.path.exists(directory):
                os.makedirs(directory)  # 创建目录
            im_alpha.save(img_out_file)

    def get_superpixel(self, tokenization_info, patch_embed, scale_factor,attn_dict_list = None,sp_shape = None, soft = False):
        assert tokenization_info is not None
        colormap = superpixel_ops.create_superpixel_colormap("random")
        res = {}
        if soft is False and attn_dict_list:
            for key, similarities in tokenization_info.items():
                if similarities.dim() == 5 and similarities.size(1) > 1:
                    # visualize each head
                    for i in range(similarities.size(1)):
                        val = prepare_similarities(
                            patch_embed,
                            similarities[:, i],
                            scale_factor,
                            merge_multihead_similarities=False,
                        )
                        labels = superpixel_ops.compute_hard_association(val).cpu()
                        # labels = superpixel_ops.compute_soft_association(val).cpu()                            
                            
                        # attn_maps = self.get_attn_maps(sp_shape, attn_dict_list)
                        for layer_idx, attn_map in enumerate(attn_dict_list):
                            if attn_map.shape[1] > 1:
                                for j in range(attn_map.shape[1]):
                                    slice_map = attn_map[:,j,...] # Extracting slice and keeping 5D tensor shape
                                    slice_map = rearrange(slice_map, 'b g (h w) -> b g h w',
                                                          h = sp_shape[-2], w = sp_shape[-1])
                                    slice_map = F.interpolate(slice_map, size=sp_shape[2:], mode='bilinear', align_corners=self.align_corners)
                                    group_result = slice_map.argmax(dim=1).cpu().numpy()
                                    rows = labels // sp_shape[-1]
                                    cols = labels % sp_shape[-2]
                                    labels_ = group_result[0, rows, cols]
                                    uni  = np.unique(labels_[0])
                                    vis = colormap[labels_ % colormap.size(0)]
                                    # import h5py
                                    # with h5py.File(f'/data2/yunfei/{key}_head{i}_{layer_idx}.h5', 'a') as hf:
                                    #     hf.create_dataset('my_dataset', data=labels)
                                    res[f"{key}_head{i}_{layer_idx}"] = vis
                similarities = prepare_similarities(
                    patch_embed, similarities, scale_factor, merge_multihead_similarities=True
                )
                labels = superpixel_ops.compute_hard_association(similarities).cpu()
                # import h5py
                # with h5py.File('embedding.h5', 'w') as f:
                #     # Create datasets for each embedding
                #     f.create_dataset('labels', data=labels.numpy())          
                # attn_maps = self.get_attn_maps(sp_shape, attn_dict_list)
                for layer_idx, attn_map in enumerate(attn_dict_list):
                    if attn_map.shape[1] > 1:
                        for j in range(attn_map.shape[1]):
                            slice_map = attn_map[:,j,...] # Extracting slice and keeping 5D tensor shape
                            slice_map = rearrange(slice_map, 'b g (h w) -> b g h w',
                                                          h = sp_shape[-2], w = sp_shape[-1])
                            slice_map = F.interpolate(slice_map, size=sp_shape[2:], mode='bilinear', align_corners=self.align_corners)
                            group_result = slice_map.argmax(dim=1).cpu().numpy()
                            rows = labels // sp_shape[-1]
                            cols = labels % sp_shape[-2]
                            labels_ = group_result[0,rows, cols]
                            vis = colormap[labels_ % colormap.size(0)]
                            
                            res[f"{key}_{layer_idx}"] = vis
            return res
        elif soft is True and attn_dict_list:
            for key, similarities in tokenization_info.items():
                if similarities.dim() == 5 and similarities.size(1) > 1:
                    # visualize each head
                    for i in range(similarities.size(1)):
                        val = prepare_similarities(
                            patch_embed,
                            similarities[:, i],
                            scale_factor = 1,
                            merge_multihead_similarities=False,
                        )
                        association_glob = superpixel_ops.compute_soft_association(val)
                        _, h, w, _, _ = association_glob.shape
                        association_glob = rearrange(
                            association_glob,
                            'b h w sh sw-> b (h w) (sh sw)'
                        )
                        
                        attn_maps = self.get_attn_maps_v2(sp_shape, attn_dict_list)
                        for layer_idx, attn_map in enumerate(attn_maps):
                            attn_map = rearrange(attn_map, 'b sh sw g -> b (sh sw) g')
                            group_result = association_glob @ attn_map
                            group_result = rearrange(
                                group_result,
                                'b (h w) g-> b g h w',
                                h = h, w = w
                            )
                            group_result = F.interpolate(group_result,size = (640,640),mode="bilinear")
                            group_result = rearrange(
                                group_result,
                                'b g h w -> b h w g',
                            )
                            group_result = group_result.argmax(dim=-1).cpu().numpy()

                            vis = colormap[group_result % colormap.size(0)]
                            # import h5py
                            # with h5py.File(f'/data2/yunfei/{key}_head{i}_{layer_idx}.h5', 'a') as hf:
                            #     hf.create_dataset('my_dataset', data=labels)
                            res[f"{key}_head{i}_group_token_{layer_idx}"] = vis
                similarities = prepare_similarities(
                    patch_embed, similarities, scale_factor = 1, merge_multihead_similarities=True
                )
                association_glob = superpixel_ops.compute_soft_association(similarities)
                _, h, w, _, _ = association_glob.shape
                association_glob = rearrange(
                    association_glob,
                    'b h w sh sw-> b (h w) (sh sw)'
                )
                attn_maps = self.get_attn_maps_v2(sp_shape, attn_dict_list)
                for layer_idx, attn_map in enumerate(attn_maps):
                    attn_map = rearrange(attn_map, 'b sh sw g -> b (sh sw) g')
                    group_result = association_glob @ attn_map
                    group_result = rearrange(
                        group_result,
                        'b (h w) g-> b g h w',
                        h = h, w = w
                    )
                    group_result = F.interpolate(group_result,size = (640,640),mode="bilinear")
                    group_result = rearrange(
                        group_result,
                        'b g h w -> b h w g',
                    )
                    group_result = group_result.argmax(dim=-1).cpu().numpy()

                    vis = colormap[group_result % colormap.size(0)]
                    # import h5py
                    # with h5py.File(f'/data2/yunfei/{key}_head{i}_{layer_idx}.h5', 'a') as hf:
                    #     hf.create_dataset('my_dataset', data=labels)
                    res[f"{key}_group_token_{layer_idx}"] = vis
            return res

        else:
            for key, similarities in tokenization_info.items():
                if similarities.dim() == 5 and similarities.size(1) > 1:
                    # visualize each head
                    for i in range(similarities.size(1)):
                        val = prepare_similarities(
                            patch_embed,
                            similarities[:, i],
                            scale_factor,
                            merge_multihead_similarities=False,
                        )
                        labels = superpixel_ops.compute_hard_association(val).cpu()
                            
                        vis = colormap[labels % colormap.size(0)]
                        
                        res[f"{key}_head{i}"] = vis
                similarities = prepare_similarities(
                    patch_embed, similarities, scale_factor, merge_multihead_similarities=True
                )
                labels = superpixel_ops.compute_hard_association(similarities).cpu()
                vis = colormap[labels % colormap.size(0)]
                res[key] = vis
            return res
    def visualize_sp_feature_v2(self, superpixel, gt):
        import h5py
        import os.path as osp
        h = w = int(math.sqrt(superpixel.shape[1]))
        superpixel = rearrange(
            superpixel,
            'b (h w) c -> b c h w',
            h = h, w=w
        )
        counter = 0  # 初始化计数器
        out_file = osp.join(self.output_dir, 'vis_sp', f'sp_{counter}.h5')

        # 检查是否存在相应的目录，不存在则创建
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))

        # 检查文件是否已存在并有超过 6 个 keys
        while True:
            out_file = osp.join(self.output_dir, 'vis_sp', f'sp_{counter}.h5')

            if not osp.exists(osp.dirname(out_file)):
                os.makedirs(osp.dirname(out_file))

            with h5py.File(out_file, 'a') as f:
                keys = list(f.keys())
                if len(keys) > 6:
                    counter += 1  # Increment counter and go to next iteration to create new file
                    continue
                
                # Process 'superpixel' data
                key = "superpixel"
                original_key = key
                count = 0
                while f"{original_key}{count}" in keys:
                    count += 1
                key = f"{original_key}{count}"
                f.create_dataset(key, data=superpixel.detach().cpu().numpy())
                
                if gt is not None:
                    # Process 'gt' data
                    h = w = int(math.sqrt(gt.shape[1]))
                    gt = rearrange(gt, 'b (h w) c -> b c h w', h=h, w=w)
                    
                    key = "gt"
                    original_key = key
                    count = 0
                    while f"{original_key}{count}" in keys:
                        count += 1
                    key = f"{original_key}{count}"
                    f.create_dataset(key, data=gt.detach().cpu().numpy())

                break  # Exit the while loop once data is written
        

    
    # def get_superpixel(self, tokenization_info, patch_embed, scale_factor,attn_dict_list = None,sp_shape = None):
    #     assert tokenization_info is not None
    #     colormap = superpixel_ops.create_superpixel_colormap("random")
    #     import h5py
    #     import os.path as osp
    #     counter = 0
    #     out_file = osp.join(self.output_dir, 'vis_association','feat.h5')
    #     directory = osp.dirname(out_file)
    #     if osp.exists(directory) is not True:
    #         os.makedirs(directory)
    #     base_name, file_ext = osp.splitext(out_file)
    #     while osp.exists(f"{base_name}_{counter}_{file_ext}"):
    #         counter += 1
    #     with h5py.File(f"{base_name}_{counter}_{file_ext}", 'a') as hf:
    #         res = {}
    #         if attn_dict_list:
    #             for key, similarities in tokenization_info.items():
    #                 if similarities.dim() == 5 and similarities.size(1) > 1:
    #                     # visualize each head
    #                     for i in range(similarities.size(1)):
    #                         val = prepare_similarities(
    #                             patch_embed,
    #                             similarities[:, i],
    #                             scale_factor,
    #                             merge_multihead_similarities=False,
    #                         )
    #                         labels = superpixel_ops.compute_hard_association(val).cpu()                            
    #                         attn_maps = self.get_attn_maps_v2(sp_shape, attn_dict_list)
    #                         for layer_idx, attn_map in enumerate(attn_maps):
    #                             attn_map = rearrange(attn_map, 'b h w g -> b g h w')
    #                             group_result = attn_map.argmax(dim=1).cpu().numpy()
    #                             rows = labels // sp_shape[-1]
    #                             cols = labels % sp_shape[-2]
    #                             labels = group_result[0, rows, cols]
    #                             uni  = np.unique(labels[0])
    #                             vis = colormap[labels % colormap.size(0)]
    #                             # import h5py
    #                             # with h5py.File(f'/data2/yunfei/{key}_head{i}_{layer_idx}.h5', 'a') as hf:
    #                             #     hf.create_dataset('my_dataset', data=labels)
    #                             res[f"{key}_head{i}_{layer_idx}"] = vis
    #                 similarities = prepare_similarities(
    #                     patch_embed, similarities, scale_factor, merge_multihead_similarities=True
    #                 )
    #                 labels = superpixel_ops.compute_hard_association(similarities).cpu()
    #                 hf.create_dataset(f"{key}_before", data=labels.detach().cpu().numpy())
                    
    #                 attn_maps = self.get_attn_maps_v2(sp_shape, attn_dict_list)
    #                 for layer_idx, attn_map in enumerate(attn_maps):
    #                     attn_map = rearrange(attn_map, 'b h w g -> b g h w')
    #                     group_result = attn_map.argmax(dim=1).cpu().numpy()
    #                     rows = labels // sp_shape[-1]
    #                     cols = labels % sp_shape[-2]

    #                     labels = group_result[0,rows, cols]
    #                     hf.create_dataset(f"group_result_layer_{layer_idx}", data=group_result)
    #                     hf.create_dataset(f"{key}_after_layer_{layer_idx}", data=labels)
    #                     vis = colormap[labels % colormap.size(0)]
    #                     res[f"{key}_{layer_idx}"] = vis
    #             return res
    #         else:
    #             for key, similarities in tokenization_info.items():
    #                 if similarities.dim() == 5 and similarities.size(1) > 1:
    #                     # visualize each head
    #                     for i in range(similarities.size(1)):
    #                         val = prepare_similarities(
    #                             patch_embed,
    #                             similarities[:, i],
    #                             scale_factor,
    #                             merge_multihead_similarities=False,
    #                         )
    #                         labels = superpixel_ops.compute_hard_association(val).cpu()                            
    #                         vis = colormap[labels % colormap.size(0)]
                            
    #                         res[f"{key}_head{i}"] = vis
    #                 similarities = prepare_similarities(
    #                     patch_embed, similarities, scale_factor, merge_multihead_similarities=True
    #                 )
    #                 labels = superpixel_ops.compute_hard_association(similarities).cpu()
    #                 vis = colormap[labels % colormap.size(0)]
    #                 res[key] = vis
    #             return res

    def get_attn_maps(self, sp_shape, attn_dict_list, return_onehot=False, rescale=False):
        """
        Args:
            img: [B, C, H, W]

        Returns:
            attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
        """
        attn_maps = []
        with torch.no_grad():
            prev_attn_masks = None
            for idx, attn_dict in enumerate(attn_dict_list):
                if attn_dict is None:
                    assert idx == len(attn_dict_list) - 1, 'only last layer can be None'
                    continue
                # [B, G, HxW]
                # B: batch size (1), nH: number of heads, G: number of group token
                # attn_masks = attn_dict['soft']
                attn_masks = attn_dict
                
                # [B, nH, G, HxW] -> [B, nH, HxW, G]
                attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
                if prev_attn_masks is None:
                    prev_attn_masks = attn_masks
                else:
                    # prev_attn_masks = prev_attn_masks @ attn_masks
                    # prev_attn_masks = torch.mul(prev_attn_masks , attn_masks)

                    prev_attn_masks = attn_masks

                # [B, nH, HxW, G] -> [B, nH, H, W, G]
                attn_maps.append(resize_attn_map(prev_attn_masks, *sp_shape[-2:]))
        length = len(attn_maps)
        i = 0
        while i < length:
            attn_map = attn_maps[i]
            # [B, nh, H, W, G]
            if attn_map.shape[1] != 1:
                new_maps = []
                for j in range(attn_map.shape[1]):
                    slice_map = attn_map[:,j,...].unsqueeze(0) # Extracting slice and keeping 5D tensor shape
                    slice_map = slice_map.squeeze(1)
                    if rescale:
                        slice_map = rearrange(slice_map, 'b h w g -> b g h w')
                        slice_map = F.interpolate(
                    slice_map, size=sp_shape[1:3], mode='bilinear', align_corners=self.align_corners)
                        slice_map = rearrange(slice_map, 'b g h w -> b h w g')
                    if return_onehot:
                        slice_map = F.one_hot(slice_map.argmax(dim=-1), num_classes=slice_map.shape[-1]).to(dtype=slice_map.dtype)
                    new_maps.append(slice_map)
                attn_maps.pop(i)  # Remove the original map
                for new_map in new_maps:
                    attn_maps.insert(i, new_map)
                
                # Jump the index over the newly added slices
                i += len(new_maps)
                length += len(new_maps) - 1  # Adjust the total length after insertion            
            else:
                attn_map = attn_map.squeeze(1)

                if rescale:
                    attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                    attn_map = F.interpolate(
                        attn_map, size=sp_shape[1:3], mode='bilinear', align_corners=self.align_corners)
                    attn_map = rearrange(attn_map, 'b g h w -> b h w g')

                if return_onehot:
                    # [B, H, W, G]
                    attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

                attn_maps[i] = attn_map
                i+=1

        return attn_maps
    def get_attn_maps_v2(self, sp_shape, attn_dict_list, return_onehot=False, rescale=False):
        """
        Args:
            img: [B, C, H, W]

        Returns:
            attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
        """
        attn_maps = []
        with torch.no_grad():
            prev_attn_masks = None
            for idx, attn_dict in enumerate(attn_dict_list):
                if attn_dict is None:
                    assert idx == len(attn_dict_list) - 1, 'only last layer can be None'
                    continue
                # [B, G, HxW]
                # B: batch size (1), nH: number of heads, G: number of group token
                # attn_masks = attn_dict['soft']
                attn_masks = attn_dict
                
                # [B, nH, G, HxW] -> [B, nH, HxW, G]
                attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
                if prev_attn_masks is None:
                    prev_attn_masks = attn_masks
                else:
                    # prev_attn_masks = prev_attn_masks @ attn_masks
                    # prev_attn_masks = torch.mul(prev_attn_masks , attn_masks)

                    prev_attn_masks = attn_masks

                # [B, nH, HxW, G] -> [B, nH, H, W, G]
                attn_maps.append(resize_attn_map(prev_attn_masks, *sp_shape[-2:]))
        length = len(attn_maps)
        i = 0
        while i < length:
            attn_map = attn_maps[i]
            # [B, nh, H, W, G]
            if attn_map.shape[1] != 1:
                attn_map = torch.sum(attn_map, dim =1, keepdim= True)/ math.sqrt( attn_map.shape[1] )
            attn_map = attn_map.squeeze(1)

            if rescale:
                attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                attn_map = F.interpolate(
                    attn_map, size=sp_shape[1:3], mode='bilinear', align_corners=self.align_corners)
                attn_map = rearrange(attn_map, 'b g h w -> b h w g')

            if return_onehot:
                # [B, H, W, G]
                attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

            attn_maps[i] = attn_map
            i+=1

        return attn_maps
    
    def blend_result(self, img, result, palette=None, out_file=None, opacity=0.5, with_bg=False):
        import mmcv
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            palette = self.PALETTE
        
        palette = np.array(palette)
        assert palette.shape[1] == 3, palette.shape
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        if with_bg:
            fg_mask = seg != 0
            img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity
        else:
            img = img * (1 - opacity) + color_seg * opacity
            # img = img
        img = img.astype(np.uint8)

        if out_file is not None:
            for B in range(img.shape[0]):
                mmcv.imwrite(img[B], out_file)

        return img
    def visualize_grouptoken(self, img,sp_shape, attn_dict_list, output_dir):
        import os.path as osp
        import os
        out_file = osp.join(output_dir, 'vis_tokens', self.vis_token, f'{self.vis_token}.jpg')
        directory = osp.dirname(out_file)
        if not osp.exists(directory):
            os.makedirs(directory)
        attn_maps = self.get_attn_maps(sp_shape, attn_dict_list) 
        num_groups = [attn_maps[layer_idx].shape[-1] for layer_idx in range(len(attn_maps))]
        for layer_idx, attn_map in enumerate(attn_maps):
            if self.vis_token == 'first_group' and layer_idx != 0:
                continue
            if self.vis_token == 'final_group' and layer_idx != len(attn_map) - 1:
                continue
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                attn_map, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            group_result = attn_map.argmax(dim=1).cpu().numpy()
            if self.vis_token == 'all_groups':
                counter = 0
                base_name, file_ext = osp.splitext(out_file)
                while osp.exists(f"{base_name}_{counter}_layer{layer_idx}{file_ext}"):
                    counter += 1

                layer_out_file = f"{base_name}_{counter}_layer{layer_idx}{file_ext}"
            else:
                layer_out_file = out_file
            
            GROUP_PALETTE = np.loadtxt('/data2/yunfei/SpformerV1/mmseg/superformer/group_palette.txt', dtype=np.uint8)[:, ::-1]
            uni = np.unique(group_result)
            self.blend_result(
                img=img.transpose(0, 2, 3, 1),
                result=group_result,
                palette=GROUP_PALETTE,
                out_file=layer_out_file,
                opacity=0.5)
            
    def visualize_grouptoken_v2(self, img, sp_shape, attn_dict_list, output_dir):
        import os.path as osp
        import os
        out_file = osp.join(output_dir, 'vis_tokens', self.vis_token, f'{self.vis_token}.jpg')
        directory = osp.dirname(out_file)
        if not osp.exists(directory):
            os.makedirs(directory)
        attn_maps = self.get_attn_maps(sp_shape, attn_dict_list) 
        num_groups = [attn_maps[layer_idx].shape[-1] for layer_idx in range(len(attn_maps))]
        for layer_idx, attn_map in enumerate(attn_maps):
            if self.vis_token == 'first_group' and layer_idx != 0:
                continue
            if self.vis_token == 'final_group' and layer_idx != len(attn_map) - 1:
                continue
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                attn_map, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            group_result = attn_map.argmax(dim=1).cpu().numpy()
            if self.vis_token == 'all_groups':
                counter = 0
                base_name, file_ext = osp.splitext(out_file)
                while osp.exists(f"{base_name}_{counter}_layer{layer_idx}{file_ext}"):
                    counter += 1

                layer_out_file = f"{base_name}_{counter}_layer{layer_idx}{file_ext}"
            else:
                layer_out_file = out_file
            
            GROUP_PALETTE = np.loadtxt('/data2/yunfei/SpformerV1/mmseg/superformer/group_palette.txt', dtype=np.uint8)[:, ::-1]
            uni = np.unique(group_result)
            self.blend_result(
                img=img.transpose(0, 2, 3, 1),
                result=group_result,
                palette=GROUP_PALETTE,
                out_file=layer_out_file,
                opacity=0.5)

    def visualize_sp_logits(self, sp_logits,output_dir):
        from PIL import Image
        import os.path as osp
        import os
        sp_logits = sp_logits.detach().cpu().numpy()
        B, C, H, W,= sp_logits.shape
        palette = superpixel_ops.create_superpixel_colormap("random")
        segmentation_images = []
        
        for b in range(B):
            seg_map = np.argmax(sp_logits[b], axis=0)
            
            seg_img = np.zeros((H, W, 3), dtype=np.uint8)
            for c in range(C):
                mask = seg_map == c
                seg_img[mask] = palette[c]
            segmentation_images.append(seg_img)    
        out_file = osp.join(output_dir, 'vis_sp_logits', 'sp_logits.jpg')
        base_name, file_ext = osp.splitext(out_file)
        counter = 0
        while osp.exists(f"{base_name}_{counter}_{file_ext}"):
            counter += 1
        img_out_file = f"{base_name}_{counter}_{file_ext}"
        directory = osp.dirname(img_out_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        Image.fromarray(segmentation_images[0]).save(img_out_file)

def get_attn_maps_v2(sp_shape, attn_dict_list, return_onehot=False, rescale=False):
    """
    Args:
        img: [B, C, H, W]

    Returns:
        attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
    """
    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for idx, attn_dict in enumerate(attn_dict_list):
            if attn_dict is None:
                assert idx == len(attn_dict_list) - 1, 'only last layer can be None'
                continue
            # [B, G, HxW]
            # B: batch size (1), nH: number of heads, G: number of group token
            # attn_masks = attn_dict['soft']
            attn_masks = attn_dict
            
            # [B, nH, G, HxW] -> [B, nH, HxW, G]
            attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                # prev_attn_masks = prev_attn_masks @ attn_masks
                # prev_attn_masks = torch.mul(prev_attn_masks , attn_masks)

                prev_attn_masks = attn_masks

            # [B, nH, HxW, G] -> [B, nH, H, W, G]
            attn_maps.append(resize_attn_map(prev_attn_masks, *sp_shape[-2:]))
    length = len(attn_maps)
    i = 0
    while i < length:
        attn_map = attn_maps[i]
        # [B, nh, H, W, G]
        if attn_map.shape[1] != 1:
            attn_map = torch.sum(attn_map, dim =1, keepdim= True)/ math.sqrt( attn_map.shape[1] )
            # new_maps = []
            # for j in range(attn_map.shape[1]):
            #     slice_map = attn_map[:,j,...].unsqueeze(0) # Extracting slice and keeping 5D tensor shape
            #     slice_map = slice_map.squeeze(1)
            #     if rescale:
            #         slice_map = rearrange(slice_map, 'b h w g -> b g h w')
            #         slice_map = F.interpolate(
            #     slice_map, size=sp_shape[1:3], mode='bilinear', align_corners=self.align_corners)
            #         slice_map = rearrange(slice_map, 'b g h w -> b h w g')
            #     if return_onehot:
            #         slice_map = F.one_hot(slice_map.argmax(dim=-1), num_classes=slice_map.shape[-1]).to(dtype=slice_map.dtype)
            #     new_maps.append(slice_map)
            # attn_maps.pop(i)  # Remove the original map
            # for new_map in new_maps:
            #     attn_maps.insert(i, new_map)
            
            # # Jump the index over the newly added slices
            # i += len(new_maps)
            # length += len(new_maps) - 1  # Adjust the total length after insertion            
        # else:
        attn_map = attn_map.squeeze(1)

        if rescale:
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                attn_map, size=sp_shape[1:3], mode='bilinear', align_corners=False)
            attn_map = rearrange(attn_map, 'b g h w -> b h w g')

        if return_onehot:
            # [B, H, W, G]
            attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

        attn_maps[i] = attn_map
        i+=1

    return attn_maps

    
        
@register_model
def asym_bottleneck_small_nofinal_head4(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)

    if pretrained:
        raise NotImplementedError()

    defaults = dict(
        sp_global_init_method="patch",
        sp_method="sp_cross_asymmetry",
        sp_features_init_methods=("from_feature", "from_feature"),
        sp_heads=(4, 4),
        sp_kwargs={
            "return_similarities_final": False,
        },
    )
    return SuperformerBottleNeck(**_update_params(defaults, **kwargs))


@register_model
def asym_bottleneck_conv4_small_nofinal_head4(**kwargs):
    return asym_bottleneck_small_nofinal_head4(
        **_update_params(dict(stem_conv_types=("conv",)), **kwargs)
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch77(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        **_update_params(
            dict(
                sp_method="patch",
                sp_global_init_method="none",
                sp_features_init_methods=("avgpool", "avgpool"),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch77_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch77(
        sp_global_init_method="lift_avgpool",
        sp_features_init_methods=("avgpool", "avgpool"),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch7(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch77(
        **_update_params(
            dict(
                depths=(12,),
                dims=(384,),
                heads=(6,),
                strides=(1,),
                sp_sizes=(4,),
                sp_heads=(1,),
                sp_features_init_methods=("avgpool",),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch7_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch7(
        **_update_params(
            dict(
                stem_channels_list=(64, 384),
                stem_conv_types=("conv", "lift_avgpool"),
                stem_kernel_sizes=(4, 4),
                stem_strides=(4, 4),
                sp_features_init_methods=("avgpool",),
                sp_sizes=(1,),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_patch7_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch7_spilavg(
        stem_kernel_sizes=(8, 4), stem_strides=(8, 4), **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_patch7_spiavgl(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_patch7_spilavg(
        stem_conv_types=("conv", "avgpool_lift"), **kwargs
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spiconv(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        sp_global_init_method="conv", **kwargs
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spiavgl(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        sp_global_init_method="avgpool_lift", **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head4_spiavgl(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spiavgl(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        **_update_params(dict(sp_global_init_method="lift_avgpool"), **kwargs)
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg_1stage(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        depths=(12,),
        dims=(384,),
        heads=(6,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("from_feature",),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg_iter1(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(sp_iter=1, **kwargs)


@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params(dict(sp_heads=(2, 2)), **kwargs)
    )

    
@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params({'sp_heads': (2, 2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150},
                         **kwargs)
    )
@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg_3stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(12,12,12),
                            'dims':(384,384,384),
                            'heads':(6,6,6),
                            'strides':(1,1,1),
                            'sp_sizes':(4,4,4),
                            'sp_heads':(1,1,1),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature")},
        **kwargs,
    )
    )

@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg_4stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2,2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(12,12,12,12),
                            'dims':(384,384,384,384),
                            'heads':(6,6,6,6),
                            'strides':(1,1,1,1),
                            'sp_sizes':(4,4,4,4),
                            'sp_heads':(2,2,2,2),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature","from_feature")},
        **kwargs,
    )
    )

@register_model
def asym_bottleneck_conv4_base_nofinal_head2_spilavg_seg_2stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(12,12),
                            'dims':(768,768),
                            'heads':(12,12),
                            'strides':(1,1,1),
                            'sp_sizes':(4,4),
                            'sp_heads':(3,3),
                            'sp_features_init_methods':("from_feature","from_feature"),
                            'stem_channels_list':(48, 96),
                            'stem_kernel_sizes':(4, 4),
                            'stem_strides':(4, 4),
                            'stem_conv_types':("conv", "conv"),
                            'pos_embed_position_method':"none"

                            },
        **kwargs,
    )
    )
@register_model
def asym_bottleneck_conv4_base_nofinal_head2_spilavg_seg_6stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(6,6,6,6,6,6,),
                            'dims':(768,768,768,768,768,768),
                            'heads':(6,6,6,6,6,6),
                            'strides':(1,1,1,1,1,1),
                            'sp_sizes':(4,4,4,4,4,4),
                            'sp_heads':(3,3,3,3,3,3),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature","from_feature","from_feature","from_feature"),
                            'stem_channels_list':(48, 96),
                            'stem_kernel_sizes':(4, 4),
                            'stem_strides':(4, 4),
                            'stem_conv_types':("conv", "conv"),
                            'pos_embed_position_method':"none"
                            },
        **kwargs,
    )
    )
@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg_9stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2,2,2,2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(6,6,6,3,3,3,2,2,2),
                            'dims':(384,384,384,384,384,384,384,384,384),
                            'heads':(6,6,6,6,6,6,6,6,6),
                            'strides':(1,1,1,1,1,1,1,1,1),
                            'sp_sizes':(4,4,4,4,4,4,4,4,4),
                            'sp_heads':(1,1,1,1,1,1,1,1,1),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature","from_feature","from_feature","from_feature","from_feature","from_feature","from_feature"),
                            'pos_embed_position_method':"none"
},
        **kwargs,
    )
    )


@register_model
def asym_bottleneck_conv4_tiny_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head2_spilavg(
        **_update_params(
            dict(
                stem_channels_list=(32,),
                dims=(192, 192),
                heads=(3, 3),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_cc4_tiny_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv4_tiny_nofinal_head2_spilavg(
        stem_kernel_sizes=(3, 3),
        stem_conv_types=("conv", "conv"),
        stem_channels_list=(16, 32),
        stem_strides=(2, 2),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_tiny_nofinal_head1_spilavg(**kwargs):
    return asym_bottleneck_conv4_tiny_nofinal_head2_spilavg(
        sp_heads=(1, 1),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head1_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(sp_heads=(1, 1), **kwargs)


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg_psim(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        use_pixel_similarities=True, **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head4_spilavg(
        **_update_params(dict(sp_heads=(2, 2)), **kwargs)
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_iter1(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head2_spilavg(sp_iter=1, **kwargs)


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_stage1(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head2_spilavg(
        depths=(12,),
        dims=(384,),
        heads=(6,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("from_feature",),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_patchbynograd(**kwargs):
    model = asym_bottleneck_conv8_small_nofinal_head2_spilavg(**kwargs)
    for stage in model.stages:
        for block in stage.patch_embed.blocks:
            for ls in [block.pixel_ls1, block.sp_ls1]:
                if hasattr(ls, "gamma"):
                    ls.gamma.requires_grad_(False)
                    ls.gamma.zero_()
                else:
                    assert isinstance(ls, nn.Identity)

    return model


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_spprojact(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head2_spilavg(
        sp_project_method="act", **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head1_spilavg(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head4_spilavg(sp_heads=(1, 1), **kwargs)


@register_model
def asym_bottleneck_conv8_small_nofinal_head2(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head4_spilavg(
        sp_heads=(2, 2), sp_global_init_method="patch", **kwargs
    )


@register_model
def asym_bottleneck_conv16_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        stem_kernel_sizes=(16,), stem_strides=(16,), **kwargs
    )


@register_model
def asym_bottleneck_cc4_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params(
            dict(
                stem_kernel_sizes=(3, 3),
                stem_conv_types=("conv", "conv"),
                stem_channels_list=(32, 64),
                stem_strides=(2, 2),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_ccc8_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params(
            dict(
                stem_kernel_sizes=(3, 3, 3),
                stem_conv_types=("conv", "conv", "conv"),
                stem_channels_list=(16, 32, 64),
                stem_strides=(2, 2, 2),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_ccc8_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_ccc8_small_nofinal_head4_spilavg(sp_heads=(2, 2), **kwargs)


@register_model
def asym_bottleneck_ccc8_small_nofinal_head2_spilavg_patchbynograd(**kwargs):
    model = asym_bottleneck_ccc8_small_nofinal_head2_spilavg(**kwargs)
    for stage in model.stages:
        for block in stage.patch_embed.blocks:
            for ls in [block.pixel_ls1, block.sp_ls1]:
                if hasattr(ls, "gamma"):
                    ls.gamma.requires_grad_(False)
                    ls.gamma.zero_()
                else:
                    assert isinstance(ls, nn.Identity)

    return model


@register_model
def asym_bottleneck_ccc8_small_nofinal_head2_spilavg77(**kwargs):
    return asym_bottleneck_ccc8_small_nofinal_head4_spilavg(
        sp_heads=(2, 2), sp_sizes=(7, 7), **kwargs
    )


@register_model
def asym_bottleneck_cc4_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_cc4_small_nofinal_head4_spilavg(
        **_update_params(dict(sp_heads=(2, 2)), **kwargs)
    )


@register_model
def asym_bottleneck_cc4_base_nofinal_head3_spilavg(**kwargs):
    return asym_bottleneck_cc4_small_nofinal_head2_spilavg(
        **_update_params(
            dict(
                stem_channels_list=(48, 96),
                dims=(768, 768),
                heads=(12, 12),
                sp_heads=(3, 3),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv4_base_nofinal_head3_spilavg(**kwargs):
    return asym_bottleneck_cc4_base_nofinal_head3_spilavg(
        **_update_params(
            dict(
                stem_channels_list=(96,),
                stem_kernel_sizes=(4,),
                stem_strides=(4,),
                stem_conv_types=("conv",),
            ),
            **kwargs,
        )
    )


# @register_model
# def asym_bottleneck_ccc8_small_nofinal_head2_spilavg(**kwargs):
#     return asym_bottleneck_cc4_small_nofinal_head2_spilavg(, **kwargs)


@register_model
def hr_bottleneck_small_nofinal(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()

    defaults = dict(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
    )
    return SuperformerBottleNeck(**_update_params(defaults, **kwargs))


@register_model
def hr_bottleneck_small_nofinal_pixelqkv(**kwargs):
    return hr_bottleneck_small_nofinal(
        sp_kwargs={
            "return_similarities_final": False,
        },
        **kwargs,
    )


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_psim(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(
        **_update_params(
            dict(
                use_pixel_similarities=True,
            ),
            **kwargs,
        )
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_psim(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv_psim(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_psim_head4(**kwargs):
    return hr_bottleneck_p8_small_nofinal_pixelqkv_psim(sp_heads=(4, 4), **kwargs)


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_head4(**kwargs):
    return hr_bottleneck_p8_small_nofinal_pixelqkv_psim_head4(
        use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_psim_head2(**kwargs):
    return hr_bottleneck_p8_small_nofinal_pixelqkv_psim(sp_heads=(2, 2), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head2(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(sp_heads=(2, 2), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head4(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(sp_heads=(4, 4), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head4_middle(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv_head4(
        use_middle_pixel_features=True, **kwargs
    )


@register_model
def hr_bottleneck_cc4_small_nofinal_pixelqkv_head4(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv_head4(
        stem_kernel_sizes=(3, 3),
        stem_conv_types=("conv", "conv"),
        stem_channels_list=(32, 64),
        stem_strides=(2, 2),
        **kwargs,
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_head4_448(**kwargs):
    assert kwargs.get("img_size") == 448, kwargs["img_size"]
    raise NotImplementedError()


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head8(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(sp_heads=(8, 8), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_patch(**kwargs):
    return hr_bottleneck_small_nofinal(
        **_update_params(
            dict(
                sp_method="patch",
                depths=(12,),
                dims=(384,),
                heads=(6,),
                strides=(1,),
                sp_sizes=(4,),
                sp_heads=(1,),
                sp_features_init_methods=("avgpool",),
            ),
            **kwargs,
        )
    )


@register_model
def hr_bottleneck_p8_small_nofinal_patch(**kwargs):
    return hr_bottleneck_small_nofinal_patch(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def hr_bottleneck_p8_small_nofinal_conv4(**kwargs):
    return hr_bottleneck_small_nofinal_patch(
        stem_kernel_sizes=(8, 4),
        stem_strides=(8, 4),
        stem_channels_list=(64, 384),
        stem_conv_types=("conv", "patch"),
        sp_sizes=(1,),
        **kwargs,
    )


@register_model
def hr_bottleneck_small_nofinal_conv4(**kwargs):
    return hr_bottleneck_small_nofinal_patch(
        stem_kernel_sizes=(4, 4),
        stem_strides=(4, 4),
        stem_channels_list=(64, 384),
        stem_conv_types=("conv", "patch"),
        sp_sizes=(1,),
        **kwargs,
    )


@register_model
def hr_bottleneck_small_nofinal_patch77(**kwargs):
    return hr_bottleneck_small_nofinal(sp_method="patch", **kwargs)


@register_model
def hr_bottleneck_p8_small_nofinal_patch77(**kwargs):
    return hr_bottleneck_small_nofinal_patch77(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table

    import models

    img_size = 224
    # img_size = 448
    model_fns = [
        models.deit_small_patch16_224_ls_1e_5,
        # hr_bottleneck_small_nofinal,
        # hr_bottleneck_p8_small_nofinal_conv4,
        # hr_bottleneck_small_nofinal_patch,
        # asym_bottleneck_small_nofinal_head4,
        # hr_bottleneck_small_nofinal_pixelqkv_head4,
        # asym_bottleneck_conv4_small_nofinal_head4_spiconv,
        # asym_bottleneck_conv4_small_nofinal_head4_spiavgl,
        # asym_bottleneck_conv4_small_nofinal_head4_spilavg,
        # models.deit_small_patch16_224,
        asym_bottleneck_conv4_small_nofinal_head2_spilavg,
        # asym_bottleneck_cc4_small_nofinal_head2_spilavg,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg,
        # asym_bottleneck_ccc8_small_nofinal_head2_spilavg,
        # models.deit_base_patch16_224,
        # models.deit_tiny_patch16_224,
        # asym_bottleneck_conv4_base_nofinal_head3_spilavg,
        # asym_bottleneck_cc4_base_nofinal_head3_spilavg,
        # asym_bottleneck_conv4_small_nofinal_patch77,
        # asym_bottleneck_conv4_small_nofinal_patch7,
        # asym_bottleneck_ccc8_small_nofinal_head2_spilavg77,
        # asym_bottleneck_conv4_small_nofinal_patch77_spilavg,
        # asym_bottleneck_conv4_small_nofinal_patch7_spilavg,
        # asym_bottleneck_conv4_small_nofinal_patch7_spilavg,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg_patchbynograd,
        # asym_bottleneck_ccc8_small_nofinal_head2_spilavg77,
        # models.deit_tiny_patch16_224,
        # asym_bottleneck_conv4_tiny_nofinal_head2_spilavg,

        # # models.deit_small_patch16_224,
        # asym_bottleneck_conv8_small_nofinal_head1_spilavg,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg_iter1,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg_stage1,
        # asym_bottleneck_conv8_small_nofinal_patch7_spilavg,
    ]

    for model_fn in model_fns:
        model = model_fn(img_size=img_size, drop_path_rate=0.1)
        # https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis
        flops = FlopCountAnalysis(model, torch.rand(1, 3, img_size, img_size))
        print(flop_count_table(flops, max_depth=3))
        # print(flop_count_table(flops, max_depth=2))

        total = sum(
            [param.nelement() for param in model.parameters() if param.requires_grad]
        )
        print("Number of parameter: %.4fM" % (total / 1e6))

    __import__("ipdb").set_trace()

    from torch.profiler import profile, record_function, ProfilerActivity

    # model = asym_bottleneck_conv4_small_nofinal_head4_spilavg().cuda()
    model = models.deit_small_patch16_224_ls_1e_5().cuda()
    inputs = torch.randn(32, 3, 224, 224).cuda()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
