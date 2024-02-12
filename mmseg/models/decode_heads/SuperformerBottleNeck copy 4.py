import math
import warnings
from collections import OrderedDict
from functools import partial
from tkinter import N
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence, Tuple, Union

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
sys.path.append("/root/autodl-tmp/GPViT/mmcls/gpvit_dev")

LayerScale2d = st.LayerScale2d
SKIP_CONFIRM = False

from models.utils.attentions import *



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


def visualize_superpixel(tokenization_info, patch_embed, scale_factor):
    assert tokenization_info is not None
    colormap = superpixel_ops.create_superpixel_colormap("random")

    res = {}
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

def build_2d_sincos_position_embedding(patches_resolution, embed_dim, temperature=10000.):
    h =  w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

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

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            timm_layers.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward_patchify(
        self, x: torch.Tensor, sp_features_last: Optional[torch.Tensor]
    ):
        if isinstance(self.patch_embed, nn.AvgPool2d):
            # NOTE(meijieru): doesn't use sp_features_last
            sp_features = self.patch_embed(x)
            return None, sp_features, x

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
        self, x: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        for i in range(start, end):
            x = self.blocks[i](x)
        return x

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
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        info, sp_features, pixel_features_middle = self.forward_patchify(
            x, sp_features_last
        )        
        assert sp_features.dim() == 4
        
        
        sp_features = self.sp_lift(sp_features)
        sp_features = sp_features.flatten(2).transpose(1, 2)  # BCHW -> BNC

        if self.cross_only == False:
            sp_features = self.add_pos_embed(sp_features)

            # sp_features = self.blocks(sp_features)

            # [0, seg_block_idx) are the blocks for segmentation
            sp_features_seg = self.forward_blocks_range(sp_features, 0, self.seg_block_idx)
            sp_features = self.forward_blocks_range(
                sp_features_seg, self.seg_block_idx, len(self.blocks)
            )

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
            )
        else:
            sp_features_seg = sp_features
            return(
                pixel_features_middle,
                sp_features,
                sp_features_seg
            )
        
            




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


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x) 
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)
        #cross attention between group token and x
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        #hard assign pixel to group token
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In GroupViT setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
    """

    def __init__(self,
                 dim,
                 out_dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 token_stride =None,
                 zero_init_group_token=False,
                 group_token_init_methods = "learnable",
                 group_token_embedding_method = "learnable",
                 merge_methods = "vanilla"
                 ):

        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        self.group_token_init_methods = group_token_init_methods
        
        if num_group_token > 0:
            if group_token_init_methods == "learnable":
                self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
                if not zero_init_group_token:
                    trunc_normal_(self.group_token, std=.02)
            
            elif group_token_init_methods == "lift_avgpool":
                norm_layer_2d = partial(get_norm_layer_2d(norm_layer))
                self.stoken_init = nn.Sequential(
                        timm_layers.create_conv2d(dim, dim, 1, padding="same"),
                        norm_layer_2d(dim),
                        nn.GELU(),
                        nn.AvgPool2d(kernel_size=token_stride, stride=token_stride),
                    )
            elif group_token_init_methods == "identity":
                self.stoken_init = nn.Identity()
        else:
            self.group_token = None
        self.merge_methods = merge_methods
        # build blocks
        
        self.depth = depth
        blocks = []
        if self.merge_methods == "vanilla":
            for i in range(depth):
                blocks.append(
                    AttnBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer))
        elif self.merge_methods == "crossfirst":
            for i in range(depth):
                blocks.append(
                    AttnBlock(
                        dim=out_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer))

        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint

        self.group_projector = group_projector
        self.group_token_embedding_method = group_token_embedding_method
        if self.group_token_embedding_method == "learnable":
            self.sp_pos_embed = nn.Parameter(torch.zeros(1,num_input_token, dim))
            self.token_pos_embed = nn.Parameter(torch.zeros(1,num_group_token,dim))
        # elif self.group_token_embedding_method == "fourier":
        #     h0 = w0 = math.sqrt(num_input_token)
        #     h1 = w1 = math.sqrt(num_group_token)
        #     sp_resolution = h0
        #     token_resolution = h1
        #     self.sp_pos_embed = build_2d_sincos_position_embedding(sp_resolution,dim)
        #     self.token_pos_embed = build_2d_sincos_position_embedding(token_resolution,dim)
        elif self.group_token_embedding_method == "fourier":
            h0 = w0 = math.sqrt(num_input_token)
            h1 = w1 = math.sqrt(num_group_token)
            sp_resolution = h0
            token_resolution = h1
            self.sp_pos_embed = build_2d_sincos_position_embedding(sp_resolution,dim)
            self.token_pos_embed = build_2d_sincos_position_embedding(token_resolution,dim)

            # import h5py
            # with h5py.File('embeddings.h5', 'w') as f:
            #     # Create datasets for each embedding
            #     f.create_dataset('sp_pos_embed', data=rearrange(self.sp_pos_embed,
            #                                                     "b  (h w) c  -> b c h w",
            #                                                     h=int(h0),w=int(h0)))
            #     f.create_dataset('token_pos_embed', data=self.token_pos_embed)
        elif self.group_token_embedding_method == "association":
            self.embedding_fn = MixerMlp(num_input_token, (num_input_token+out_dim)//2, out_dim)
                


    @property
    def with_group_token(self):
        # return self.group_token is not None
        return True

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.merge_methods == "vanilla":
            if self.group_token_init_methods == "learnable":
                group_token = self.group_token.expand(x.size(0), -1, -1) #group_token B,64,384
                if self.group_projector is not None:
                    group_token = group_token + self.group_projector(prev_group_token)
                    
            else:
                b, n, c = x.shape
                h_w = int(math.sqrt(n))
                group_token = self.stoken_init(rearrange(x, 'b (h w) c -> b c h w', h=h_w, w=h_w))
                group_token = rearrange(
                    group_token,
                    'b c h w -> b (h w) c',
                )
                
            if self.group_token_embedding_method:
                x = x +self.sp_pos_embed
                group_token = group_token +self.token_pos_embed
            B, L, C = x.shape
            cat_x = self.concat_x(x, group_token)#concat group_token and x(image token) then pass through ViT blocks
            for blk_idx, blk in enumerate(self.blocks):
                if self.use_checkpoint:
                    cat_x = checkpoint.checkpoint(blk, cat_x)
                else:
                    cat_x = blk(cat_x)

            x, group_token = self.split_x(cat_x)

            attn_dict = None
            if self.downsample is not None:
                #downsample: 1. cross attention between x and group_token
                #2.assign x to group token
                x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)

            return x, group_token, attn_dict
        elif self.merge_methods == "crossfirst":
            if self.group_token_init_methods == "learnable":
                group_token = self.group_token.expand(x.size(0), -1, -1) #group_token B,64,384
                if self.group_projector is not None:
                    group_token = group_token + self.group_projector(prev_group_token)
            attn_dict = None
            if self.downsample is not None:
                #downsample: 1. cross attention between x and group_token
                #2.assign x to group token
                x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)
            if self.group_token_embedding_method == "association":
                token_pos_embed = self.embedding_fn(
                    attn_dict['soft'].squeeze(1).transpose(1,2)).transpose(1,2)
                x = x + token_pos_embed
            elif self.group_token_embedding_method == "fourier":
                x = x + self.token_pos_embed
            for blk_idx, blk in enumerate(self.blocks):
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            return x, group_token, attn_dict

        elif self.merge_methods == "crossfirst_noblock":
            if self.group_token_init_methods == "learnable":
                group_token = self.group_token.expand(x.size(0), -1, -1) #group_token B,64,384
                if self.group_projector is not None:
                    group_token = group_token + self.group_projector(prev_group_token)
            attn_dict = None
            if self.group_token_embedding_method == "fourier":
                x = x +self.sp_pos_embed
                group_token = group_token +self.token_pos_embed

            if self.downsample is not None:
                #downsample: 1. cross attention between x and group_token
                #2.assign x to group token
                x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)                
            return x, group_token, attn_dict
class GPBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 depth,
                 num_group_heads,
                 num_ungroup_heads,
                 num_group_token,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 fwd_att_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 **kwargs):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        self.group_token = nn.Parameter(torch.zeros(1, num_group_token, embed_dims))
        trunc_normal_(self.group_token, std=.02)

        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=0.,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=embed_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path=drop_path)
        _mixer_cfg.update(fwd_att_cfg)
        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

        self.dwconv = torch.nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3,3), padding=(1,1), bias=False, groups=embed_dims),
            nn.BatchNorm2d(num_features=embed_dims),
            nn.ReLU(True))

    def forward(self, x, hw_shape):
        """
        Args:
            x: image tokens, shape [B, L, C]
            hw_shape: tuple or list (H, W)
        Returns:
            proj_tokens: shape [B, L, C]
        """
        B, L, C = x.size()
        group_token = self.group_token.expand(x.size(0), -1, -1)
        gt = group_token

        gt = self.group_layer(query=gt, key=x, value=x)
        gt = self.mixer(gt)
        ungroup_tokens = self.un_group_layer(query=x, key=gt, value=gt)
        ungroup_tokens = ungroup_tokens.permute(0,2,1).contiguous().reshape(B, C, hw_shape[0], hw_shape[1])
        proj_tokens = self.dwconv(ungroup_tokens).view(B, C, -1).permute(0,2,1).contiguous().view(B, L, C)
        return proj_tokens

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
        **kwargs

    ):
        super().__init__(in_channels=3,
        channels=256,
        num_classes=20,
        out_channels=20,
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

        num_stages = len(depths)
        stages = []


        stage_in_dim = stem_channels_list[-1]

        for i, (depth, dim, head, sp_size, sp_head, stride) in enumerate(
            zip(depths, dims, heads, sp_sizes, sp_heads, strides)
        ):
            cur_stride *= stride
            sp_stride = sp_size * stride // sp_sizes[i - 1] if i > 0 else 1
            sp_layer, sp_shape = self._make_superpixel_layer(
                i, cur_stride, sp_size, sp_head, pixel_dim, dim, sp_method
            )

            # first feature already has norm & act
            pre_norm_pixel_stage = pre_norm_pixel and i != 0
            if pre_norm_pixel_stage:
                print(f"pre_norm_pixel for stage {i}")

            return_updated_pixel_features = i < num_stages - 1
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
        

        if weight_init != "skip":
            self.init_weights(weight_init)
        
        
        #downsample module from groupvit
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate_merge, sum(depths_merge))]
        self.vis_token = vis_token
        self.output_dir = output_dir
        self.vis_featuremap = vis_featuremap
        self.vis_sp_logits = vis_sp_logits

        # build layers
        self.merge_methods = merge_methods
        if self.merge_methods == 'GPViT':
            self.arch_settings = {
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
            }
            self.num_layers = self.arch_settings['num_layers']

            self.mergelayers = nn.ModuleList()
            import copy
            _arch_settings = copy.deepcopy(self.arch_settings)
            for i in range(self.num_layers):
                _layer_cfg = dict(
                        embed_dims=sp_dim,
                        depth=_arch_settings['mlpmixer_depth'],
                        num_group_heads=_arch_settings['num_group_heads'],
                        num_forward_heads=_arch_settings['num_group_forward_heads'],
                        num_ungroup_heads=_arch_settings['num_ungroup_heads'],
                        num_group_token=_arch_settings['group_layers'][i],
                        ffn_ratio=_arch_settings['ffn_ratio'],
                        drop_path=dpr[i],
                        with_cp=None)
                group_layer = GPBlock(**_layer_cfg)
                self.mergelayers.append(group_layer)
        else:
            self.mergelayers = nn.ModuleList()
            self.num_merges = len(num_group_tokens)
            self.group_token_init_methods = group_token_init_methods
            if group_token_init_methods == "from_pixel":
                token_size = int(math.sqrt(num_group_tokens[0]))
                token_dim = token_channels_list[0]
                self.token_init = nn.Sequential(
                    timm_layers.create_conv2d(pixel_dim, token_dim, 1, padding="same"),
                    norm_layer_2d(token_dim),
                    nn.GELU(),
                    nn.AvgPool2d(kernel_size=token_size, stride=token_size),
                )
            for i_layer in range(self.num_merges):
                num_input_token = sp_shape[0]* sp_shape[1]
                if i_layer == 0:
                    dim = dims[-1]
                else:
                    dim = token_channels_list[i_layer-1]
                downsample = None
                out_dim = token_channels_list[i_layer]

                if i_layer < self.num_merges :
                    downsample = GroupingBlock(
                        dim=dim,
                        out_dim=out_dim,
                        num_heads=merge_heads[i_layer],
                        num_group_token=num_group_tokens[i_layer],
                        num_output_group=num_output_groups[i_layer],
                        norm_layer=norm_layer,
                            hard=hard_assignment[i_layer],
                            gumbel=hard_assignment[i_layer],)

                    num_output_token = num_output_groups[i_layer]

                self.group_projector_from_sp_feature = group_projector_from_sp_feature
                if num_group_tokens[i_layer] > 0:

                    if i_layer == 0:
                        if group_projector_from_sp_feature:
                            prev_dim = int(dims[-1])
                            group_projector = nn.Sequential(
                            norm_layer(prev_dim),
                            MixerMlp(num_input_token, prev_dim // 2, num_group_tokens[i_layer]))
                        else:
                            group_projector = None
                    elif i_layer == 1:
                        prev_dim = int(dims[-1])
                        group_projector = nn.Sequential(
                            norm_layer(prev_dim),
                            MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))
                        if dim != prev_dim:
                            group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                    nn.Linear(prev_dim, dim, bias=False))

                    else:
                        prev_dim = int(token_channels_list[i_layer-1])
                        group_projector = nn.Sequential(
                            norm_layer(prev_dim),
                            MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))
                        if dim != prev_dim:
                            group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                        nn.Linear(prev_dim, dim, bias=False))

                else:
                    group_projector = None
                layer = GroupingLayer(
                    dim=dim,
                    out_dim=out_dim,
                    num_input_token=num_input_token if i_layer == 0 else num_group_tokens[i_layer-1],
                    depth=depths_merge[i_layer],
                    num_heads=merge_heads[i_layer],
                    num_group_token=num_group_tokens[i_layer],
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=drop_rate_merge,
                    attn_drop=attn_drop_rate_merge,
                    drop_path=dpr[sum(depths_merge[:i_layer]):sum(depths_merge[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=downsample,
                    use_checkpoint=None,
                    group_projector=group_projector,
                    token_stride=token_strides[i_layer],
                    group_token_init_methods=group_token_init_methods[i_layer],
                    group_token_embedding_method = group_token_embedding_method[i_layer],
                    # only zero init group token if we have a projection
                    zero_init_group_token=group_projector is not None,
                    merge_methods=merge_methods)
                self.mergelayers.append(layer)
                if i_layer < self.num_merges - 1:
                    num_input_token = num_output_token

            self.mergelayers.apply(self.init_weights_mergelayers)
    def init_weights_mergelayers(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    
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
        if self.merge_methods=="GPViT":
            if self.vis_featuremap == True:
                import h5py
                import os.path as osp
                import os
                out_file = osp.join(self.output_dir, 'vis_featuremap','featuremap.h5')

                directory = osp.dirname(out_file)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                counter = 0
                base_name, file_ext = osp.splitext(out_file)
                while osp.exists(f"{base_name}_{counter}_{file_ext}"):
                    counter += 1

                with h5py.File(f"{base_name}_{counter}_{file_ext}", 'w') as hf:

                    for i, stage in enumerate(self.stages):
                        pixel_features, sp_features, sp_features_seg = stage(
                            pixel_features,
                            sp_features_last,
                        )                
                        sp_features_last = sp_features
                        if sp_features_last.dim() == 3:
                            b, n, c= sp_features_last.shape

                            h = w = int(math.sqrt(n))
                            
                            sp_featuremap = rearrange(
                                sp_features_last,
                                'b (h w) c -> b c h w',
                                h=h,w=w,
                            )
                        elif sp_features_last.dim() == 4:
                            sp_featuremap = sp_features_last
                        key = f"superpixel_stage{i}"

                        hf.create_dataset(key, data=sp_featuremap.detach().cpu().numpy())

                    for i, mergelayer in enumerate(self.mergelayers):
                        b, n, c = sp_features_last.shape
                        hw_shape = to_2tuple(int(math.sqrt(n)))
                        sp_features_last= mergelayer(
                            sp_features_last,hw_shape
                        )
                        sp_featuremap = rearrange(
                                sp_features_last,
                                'b (h w) c -> b c h w',
                                h=hw_shape[0],w=hw_shape[1],
                            )
                        key = f"merge_stage{i}"

                        hf.create_dataset(key, data=sp_featuremap.detach().cpu().numpy())

                return sp_features,sp_features_last,endpoints
            else:                    
                for i, stage in enumerate(self.stages):
                        pixel_features, sp_features, sp_features_seg = stage(
                            pixel_features,
                            sp_features_last,
                        )                
                        sp_features_last = sp_features
                for i, mergelayer in enumerate(self.mergelayers):
                    b, n, c = sp_features_last.shape
                    hw_shape = to_2tuple(int(math.sqrt(n)))
                    sp_features_last= mergelayer(
                        sp_features_last,hw_shape
                    )
                return sp_features,sp_features_last,endpoints

                
        else:
            attn_dict_list = []
            if self.vis_featuremap == True:
                import h5py
                import os.path as osp
                import os
                out_file = osp.join(self.output_dir, 'vis_featuremap','featuremap.h5')

                directory = osp.dirname(out_file)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                counter = 0
                base_name, file_ext = osp.splitext(out_file)
                while osp.exists(f"{base_name}_{counter}_{file_ext}"):
                    counter += 1

                with h5py.File(f"{base_name}_{counter}_{file_ext}", 'w') as hf:
                    for i, stage in enumerate(self.stages):
                        pixel_features, sp_features, sp_features_seg = stage(
                            pixel_features,
                            sp_features_last,
                        )                
                        sp_features_last = sp_features
                        if sp_features_last.dim() == 3:
                            b, n, c= sp_features_last.shape
                            h = w = int(math.sqrt(n))
                            
                            sp_featuremap = rearrange(
                                sp_features_last,
                                'b (h w) c -> b c h w',
                                h=h,w=w,
                            )
                        elif sp_features_last.dim() == 4:
                            sp_featuremap = sp_features_last
                        key = f"superpixel_stage{i}"

                        hf.create_dataset(key, data=sp_featuremap.detach().cpu().numpy())
                        

                    

                # # rank not consistent due to whether flatten or not
                # # res[f"sp_features_stage{i}"] = sp_features
                # if return_updated_pixel_features:
                #     endpoints[f"pixel_features_stage{i}"] = pixel_features
                # downsample by using group token
                    grouped_img_tokens = sp_features_last
                    group_token = grouped_img_tokens
    
                    for i, mergelayer in enumerate(self.mergelayers):
                        if i < self.num_merges:
                            grouped_img_tokens, group_token, attn_dict = mergelayer(
                                grouped_img_tokens, group_token, return_attn=True
                            )
                            key = f"grouptoken_stage{i}"
                            b, n, c= grouped_img_tokens.shape
                            h = w = int(math.sqrt(n))
                            hf.create_dataset(key, data=rearrange(grouped_img_tokens,
                                                                'b (h w) c-> b c h w',
                                                                h=h,w=w).detach().cpu().numpy())

                            attn_dict_list.append(attn_dict)
            else:
                for i, stage in enumerate(self.stages):
                    pixel_features, sp_features, sp_features_seg = stage(
                        pixel_features,
                        sp_features_last,
                    )                
                    sp_features_last = sp_features
                
                # if self.group_token_init_methods=="from_pixel":
                #     grouped_img_tokens = self.init_superpixel_features(pixel_features)
                    
                # after SuperformerStage we use mergelayer for group token generating and merging
                # grouped_img_tokens:image tokens(initiate as superpixel feature)       
                
                grouped_img_tokens = sp_features_last
                if self.group_projector_from_sp_feature == True:
                    group_token = grouped_img_tokens
                else:
                    group_token = None
    
                for i, mergelayer in enumerate(self.mergelayers):
                    if i < self.num_merges:
                        grouped_img_tokens, group_token, attn_dict = mergelayer(
                            grouped_img_tokens, group_token, return_attn=True
                        )
                        attn_dict_list.append(attn_dict)


            return sp_features, sp_features_seg, group_token, attn_dict_list, endpoints
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
        self, x: torch.Tensor, grouped_img_tokens: torch.Tensor=None,attn_dict_list=None, return_pixel_logits: bool = True, stride: int = 2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.merge_methods == "GPViT":
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

        else:
            grouped_img_tokens = self.seg_norm(grouped_img_tokens)
            
            b, num, c = grouped_img_tokens.shape
            if not self.seg_specific_classifier:
                raise ValueError("No segmentation head is found.")
            elif self.seg_specific_classifier=="Linear":
                
                grouped_img_tokens = grouped_img_tokens.reshape(b * num, c)

                group_token_logits = self.seg_head(grouped_img_tokens)
                group_token_logits = group_token_logits.reshape(b, num, -1)
            elif self.seg_specific_classifier=="Conv":
                grouped_img_tokens = rearrange(
                    grouped_img_tokens,
                    'b h c -> b c h'
                )
                grouped_img_tokens = grouped_img_tokens.unsqueeze(2)
                group_token_logits = self.seg_head(grouped_img_tokens)
                group_token_logits = rearrange(
                    group_token_logits,
                    'b c h w -> b (h w) c'
                )
            
            B, N, C = x.shape
            pw = ph = int(math.sqrt(N))
            sp_shape = (B, C, ph, pw)


            # onehot_attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)
            attn_map = self.get_attn_maps(sp_shape, attn_dict_list)[-1]
            num_classes = 150
            class_offset = 0
            # pred_logits = torch.zeros(num_classes, *attn_map.shape[:2], device=x.device, dtype=x.dtype)
            onehot_attn_map = rearrange(
                attn_map,
                'b h w c -> b (h w) c')
            

            sp_logits = onehot_attn_map @ group_token_logits
            
            
            
            
            last_stage = self.stages[-1]
            last_sp_layer = last_stage.patch_embed

            if isinstance(last_sp_layer, nn.AvgPool2d):
                sh=sw = last_sp_layer.kernel_size
            elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                sh, sw = last_sp_layer.superpixel_shape
            else:
                raise ValueError()
            sp_logits = sp_logits.view(b, sh, sw, -1).permute(0, 3, 1, 2)
            if self.vis_sp_logits == True:
                self.visualize_sp_logits(sp_logits,self.output_dir)


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

    def forward(
        self,
        x: torch.Tensor,
        generate_seg: bool = True,
        return_pixel_logits: bool = True,
        seg_stride: int = 1,

    ) -> Union[torch.Tensor, MutableMapping[str, torch.Tensor]]:

        if self.merge_methods =="GPViT":
             sp_features, sp_features_seg,endpoints = self.forward_features(x)
             if generate_seg:
                sp_logits, pixel_logits = self.forward_segmentation(
                sp_features_seg, return_pixel_logits, stride=seg_stride
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
            self.visualize_grouptoken(img, sp_shape, attn_dict_list, self.output_dir)
            
            self.visualize_superpixel(img, resize_similarities= True, output_dir= self.output_dir)
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

    def visualize_superpixel(self, img, resize_similarities: bool = True, output_dir: str = None):
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
            res_i = visualize_superpixel(tokenization_info, patch_embed, scale_factor)
            res.update({f"stage{i}_{key}": val for key, val in res_i.items()})
        

        counter = 0
        base_name, file_ext = osp.splitext(out_file)
        for key, val in res.items():
            
            im = val[0].squeeze().numpy().astype(np.uint8)
            resize_output = not resize_similarities or im.shape[0] != ih
            im = Image.fromarray(im)
            if resize_output:
                print('resize_output')
                im = im.resize((ih, iw), Image.Resampling.NEAREST)
                
            alpha = 0.5
            im_alpha = np.array(im_image, dtype=np.float32) * alpha + np.array(im, dtype=np.float32) * (1 - alpha)
            im_alpha = Image.fromarray(im_alpha.astype(np.uint8))
            while osp.exists(f"{base_name}_{counter}_vis_{i}_{key}_{file_ext}"):
                counter += 1
            img_out_file = f"{base_name}_{counter}_vis_{i}_{key}_{file_ext}"
            



            im_alpha.save(img_out_file)

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
                attn_masks = attn_dict['soft']
                # [B, nH, G, HxW] -> [B, nH, HxW, G]
                attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
                if prev_attn_masks is None:
                    prev_attn_masks = attn_masks
                else:
                    prev_attn_masks = prev_attn_masks @ attn_masks  
                    # prev_attn_masks = attn_masks

                # [B, nH, HxW, G] -> [B, nH, H, W, G]
                attn_maps.append(resize_attn_map(prev_attn_masks, *sp_shape[-2:]))

        for i in range(len(attn_maps)):
            attn_map = attn_maps[i]
            # [B, nh, H, W, G]
            assert attn_map.shape[1] == 1
            # [B, H, W, G]
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
        out_file = osp.join(output_dir, 'vis_tokens', self.vis_token, f'{self.vis_token}.jpg')
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
            
            GROUP_PALETTE = np.loadtxt('/root/autodl-tmp/SpformerV1/mmseg/superformer/group_palette.txt', dtype=np.uint8)[:, ::-1]
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
