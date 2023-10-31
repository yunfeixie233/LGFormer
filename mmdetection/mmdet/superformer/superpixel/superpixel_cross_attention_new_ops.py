import math
from functools import partial
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.layers as tml

from . import superpixel_ops
from .dual_path_transformer_ops import (
    BaseModule,
    Conv2D,
    reshape_and_transpose_for_attention_operation,
)


ThreeInt = Tuple[int, int, int]


class Reweight(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.reweight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (0.5 + torch.sigmoid(self.reweight))
    
class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class DualPathCrossAttentionLayer(BaseModule):
    """Applies a transformer layer, as proposed in MaX-DeepLab models."""

    def __init__(
        self,
        in_channels_pixel: int,
        in_channels_sp: int,
        pixel_qkv_channels: ThreeInt,
        sp_qkv_channels: ThreeInt,
        num_heads_pixel: int = 1,
        num_heads_sp: int = 1,
        norm_layer=partial(tml.LayerNorm2d, eps=1e-6),
        pixel_shape=None,
        superpixel_shape=None,
        position_embedding_method="none",
        position_embedding_stride=4,
        sp_position_embedding_stride=None,
        pixel_qkv_method: str = "linear",
        ls_init_value: Optional[float] = None,
        pixelify: bool = True,
        always_return_similarity_pixel: bool = False,
        hard_assign: bool = True,
        reweight_sp: bool = False,
        reweight_pixel: bool = False,
        
    ):
        """Initializes a DualPathTransformerSimpleLayer."""
        super().__init__()

        sp_q_channels, sp_k_channels, sp_v_channels = sp_qkv_channels
        pixel_q_channels, pixel_k_channels, pixel_v_channels = pixel_qkv_channels

        assert sp_q_channels == pixel_k_channels
        assert pixel_q_channels == sp_k_channels
        assert in_channels_sp == pixel_v_channels
        assert in_channels_pixel == sp_v_channels

        if pixel_q_channels % num_heads_pixel:
            raise ValueError("Total_key_depth should be divisible by num_heads.")

        if sp_q_channels % num_heads_sp:
            raise ValueError("Total_value_depth should be divisible by num_heads.")

        self.sp_qkv_channels = sp_qkv_channels
        self.pixel_qkv_channels = pixel_qkv_channels

        # Compute query key value with one convolution and a batch norm layer. The
        # initialization std is standard transformer initialization (without batch
        # norm), as used in SASA and ViT. In our case, we use batch norm by default,
        # so it does not require careful tuning. If one wants to remove all batch
        # norms in axial attention, this standard initialization should still be
        # good, but a more careful initialization is encouraged.
        # FIXME(meijieru): initialization
        # initialization_std = bottleneck_channels**-0.5

        # TODO(meijieru): if we don't need pixelify, we can further remove sp_value
        self._sp_qkv = Conv2D(
            in_channels_sp,
            sum(sp_qkv_channels),
            use_bn=True,
            bn_layer=norm_layer,
            activation="none",
        )

        self._always_return_similarity_pixel = always_return_similarity_pixel
        self._pixelify = pixelify
        self._pixel_qkv_method = pixel_qkv_method
        if self._pixel_qkv_method == "linear":
            self._pixel_qkv = Conv2D(
                in_channels_pixel,
                sum(pixel_qkv_channels),
                use_bn=True,
                bn_layer=norm_layer,
                activation="none",
            )
        else:
            raise ValueError(f"Unknown pixel_qkv_method: {self._pixel_qkv_method}")

        self._num_heads_pixel = num_heads_pixel
        self._num_heads_sp = num_heads_sp
        self._position_embedding_method = position_embedding_method
        self._position_embedding_stride = position_embedding_stride
        self._sp_position_embedding_stride = (
            sp_position_embedding_stride or position_embedding_stride
        )

        if self._position_embedding_method == "none":
            pass
        elif self._position_embedding_method == "learnable":
            self.sp_pos_embed = self._create_pos_embed(
                in_channels_sp, superpixel_shape, self._sp_position_embedding_stride
            )
            self.pixel_pos_embed = self._create_pos_embed(
                in_channels_pixel, pixel_shape, self._position_embedding_stride
            )
            print(
                f"sp_pos_embed shape: {self.sp_pos_embed.shape}, "
                f"pixel_pos_embed shape: {self.pixel_pos_embed.shape}"
            )
        elif self._position_embedding_method == "depthwise":
            self.sp_pos_conv = tml.create_conv2d(
                in_channels_sp, in_channels_sp, 3, bias=True, depthwise=True
            )
            self.pixel_pos_conv = tml.create_conv2d(
                in_channels_pixel,
                in_channels_pixel,
                3,
                bias=True,
                depthwise=True,
            )
        else:
            raise ValueError()

        if ls_init_value is not None:
            self.sp_ls1 = LayerScale2d(in_channels_sp, ls_init_value)
            if pixelify:
                self.pixel_ls1 = LayerScale2d(in_channels_pixel, ls_init_value)
            else:
                self.pixel_ls1 = nn.Identity()
        else:
            self.sp_ls1 = nn.Identity()
            self.pixel_ls1 = nn.Identity()
            
        
        self.reweight_sp = Reweight() if reweight_sp else nn.Identity()
        self.reweight_pixel = Reweight() if reweight_pixel else nn.Identity()
        self.hard_assign = hard_assign
    def _create_pos_embed(self, dim, shape, stride) -> torch.Tensor:

        pos_embed_shape = [int(math.ceil(val / stride)) for val in shape]
        for val in pos_embed_shape:
            assert val > 1
        pos_embed = nn.Parameter(torch.zeros(1, dim, *pos_embed_shape))
        return pos_embed

    def _add_pos_embed(self, val, pos_embed) -> torch.Tensor:
        _, _, h, w = val.shape
        return val + F.interpolate(
            pos_embed, size=(h, w), mode="bilinear", align_corners=False
        )

    def _split_qkv(
        self,
        val: torch.Tensor,
        qkv_channels: Sequence[int],
        qkv_method: str,
    ) -> Sequence[torch.Tensor]:
        if qkv_method == "linear":
            query, key, value = torch.split(
                val,
                list(qkv_channels),
                dim=1,
            )
        else:
            raise ValueError(f"Unknown pixel_qkv_method: {self._pixel_qkv_method}")

        return query, key, value

    def _superpixel_update_step(
        self,
        sp_query: torch.Tensor,
        pixel_key: torch.Tensor,
        pixel_value: torch.Tensor,
        
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sp_query, pixel_key, pixel_value = [
            reshape_and_transpose_for_attention_operation(val, self._num_heads_sp)
            for val in [sp_query, pixel_key, pixel_value]
        ]

        bsp, _, _, sh, sw = sp_query.shape
        b, num_heads, c_key, h, w = pixel_key.shape
        _, _, c_value, _, _ = pixel_value.shape
        assert b == bsp
        scale = c_key**-0.5
        # [b * num_heads]
        sp_query, pixel_key, pixel_value = [
            val.flatten(end_dim=1) for val in [sp_query, pixel_key, pixel_value]
        ]
        similarities = (
            superpixel_ops.compute_similarities_dot_product_sp(pixel_key, sp_query) * scale
        )

        sp_delta = superpixel_ops.update_superpixel_features(
            pixel_value, sp_query, similarities,hard_assign=self.hard_assign
        )
        sp_delta = sp_delta.reshape(b, num_heads * c_value, sh, sw)
        return sp_delta, similarities.reshape(b, num_heads, 9, h, w)

    def _superpixel_assign_step(
        self,
        pixel_query: torch.Tensor,
        sp_key: torch.Tensor,
        sp_value: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._always_return_similarity_pixel and not self._pixelify:
            return None, None

        pixel_query, sp_key, sp_value = [
            reshape_and_transpose_for_attention_operation(val, self._num_heads_pixel)
            for val in [pixel_query, sp_key, sp_value]
        ]

        bsp, _, _, _, _ = sp_key.shape
        b, num_heads, c_key, h, w = pixel_query.shape
        _, _, c_value, _, _ = sp_value.shape
        assert b == bsp
        scale = c_key**-0.5
        pixel_query, sp_key, sp_value = [
            val.flatten(end_dim=1) for val in [pixel_query, sp_key, sp_value]
        ]
        similarities = (
            superpixel_ops.compute_similarities_dot_product_sp(pixel_query, sp_key) * scale
        )
        pixel_delta = None
        if self._pixelify:
            pixel_delta = superpixel_ops.update_pixel_features(
                None, sp_value, similarities
            )
            pixel_delta = pixel_delta.reshape(b, num_heads * c_value, h, w)
        return pixel_delta, similarities.reshape(b, num_heads, 9, h, w)

    def forward(self, pixel_features, sp_features):
        """Performs a forward pass."""
        # [b, c, h, w], [b, csp, sh, sw]

        if self._position_embedding_method == "learnable":
            sp_features_embeded = self._add_pos_embed(sp_features, self.sp_pos_embed)
            pixel_features_embeded = self._add_pos_embed(
                pixel_features, self.pixel_pos_embed
            )
        elif self._position_embedding_method == "depthwise":
            sp_features_embeded = self.sp_pos_conv(sp_features) + sp_features
            pixel_features_embeded = (
                self.pixel_pos_conv(pixel_features) + pixel_features
            )

        else:
            sp_features_embeded = sp_features
            pixel_features_embeded = pixel_features

        sp_query, sp_key, sp_value = self._split_qkv(
            self._sp_qkv(sp_features_embeded), self.sp_qkv_channels, "linear"
        )

        pixel_query, pixel_key, pixel_value = self._split_qkv(
            self._pixel_qkv(pixel_features_embeded),
            self.pixel_qkv_channels,
            self._pixel_qkv_method,
        )

        (
            sp_feature_delta,
            similarities_multi_head,
        ) = self._superpixel_update_step(sp_query, pixel_key, pixel_value)

        (
            pixel_features_delta,
            similarities_multi_head_pixel,
        ) = self._superpixel_assign_step(pixel_query, sp_key, sp_value)
        if self._pixelify:
            pixel_return =  self.reweight_pixel(pixel_features) + self.pixel_ls1(pixel_features_delta)
        else:
            pixel_return = None

        return (
            pixel_return,
            self.reweight_sp(sp_features) + self.sp_ls1(sp_feature_delta),
            similarities_multi_head,
            similarities_multi_head_pixel,
        )

    def init_weights(self):
        initialization_std = 0.02
        for conv in [
            self._sp_qkv.conv,
            self._pixel_qkv.conv,
        ]:
            # NOTE(meijieru): don't use conv init, as they are actually 3 linear
            # Linear initialization like timm
            tml.trunc_normal_(conv.weight, std=initialization_std)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)