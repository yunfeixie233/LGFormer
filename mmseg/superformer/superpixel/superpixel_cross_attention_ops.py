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


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class LayerNorm2DChannelOnly(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert len(self.normalized_shape) == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [b, h, w, c]
        assert x.ndim == 4
        res = super().forward(x.permute(0, 2, 3, 1))
        return res.permute(0, 3, 1, 2)


class DualPathTransformerSimpleLayer(BaseModule):
    """Applies a transformer layer, as proposed in MaX-DeepLab models."""

    def __init__(
        self,
        in_features: int,
        filters=128,
        num_heads=2,
        key_expansion=1,
        value_expansion=1,
        norm_layer=partial(LayerNorm2DChannelOnly, eps=1e-6),
        conv_kernel_weight_decay=0.0,
        pixel_shape=None,
        superpixel_shape=None,
        position_embedding_method="none",
        position_embedding_stride=4,
        sp_position_embedding_stride=None,
        pixel_qkv_method: str = "linear",
        ls_init_value: Optional[float] = None,
        pixelify: bool = True,
        always_return_similarity_pixel: bool = True,
        temperature_method: str = "none",
    ):
        """Initializes a DualPathTransformerSimpleLayer."""
        super().__init__()

        total_key_depth = int(round(filters * key_expansion))
        total_value_depth = int(round(filters * value_expansion))

        if total_key_depth % num_heads:
            raise ValueError("Total_key_depth should be divisible by num_heads.")

        if total_value_depth % num_heads:
            raise ValueError("Total_value_depth should be divisible by num_heads.")

        # Compute query key value with one convolution and a batch norm layer. The
        # initialization std is standard transformer initialization (without batch
        # norm), as used in SASA and ViT. In our case, we use batch norm by default,
        # so it does not require careful tuning. If one wants to remove all batch
        # norms in axial attention, this standard initialization should still be
        # good, but a more careful initialization is encouraged.
        # FIXME(meijieru): initialization
        # initialization_std = bottleneck_channels**-0.5

        # TODO(meijieru): tf initialization.
        # TODO(meijieru): no qkv_bias here, check later
        self._sp_qkv = Conv2D(
            in_features,
            total_key_depth * 2 + total_value_depth,
            use_bn=True,
            bn_layer=norm_layer,
            activation="none",
        )

        # TODO(meijieru): if we don't need pixelify, we can further remove sp_value
        self._always_return_similarity_pixel = always_return_similarity_pixel
        self._pixelify = pixelify
        self._pixel_qkv_method = pixel_qkv_method
        if self._pixel_qkv_method == "linear":
            self._pixel_qkv = Conv2D(
                in_features,
                total_key_depth * 2 + total_value_depth,
                use_bn=True,
                bn_layer=norm_layer,
                activation="none",
            )
        elif self._pixel_qkv_method in ["linear_qk", "linear_qk_rawv"]:
            self._pixel_qkv = Conv2D(
                in_features,
                total_key_depth * 2,
                use_bn=True,
                bn_layer=norm_layer,
                activation="none",
            )
        elif self._pixel_qkv_method == "linear_shared":
            if total_key_depth != total_value_depth:
                raise ValueError("total_key_depth != total_value_depth")
            self._pixel_qkv = Conv2D(
                in_features,
                total_value_depth,
                use_bn=True,
                bn_layer=norm_layer,
                activation="none",
            )
        elif self._pixel_qkv_method == "identity":
            self._pixel_qkv = nn.Identity()
        elif self._pixel_qkv_method == "norm":
            self._pixel_qkv = norm_layer(in_features)
        else:
            raise ValueError(f"Unknown pixel_qkv_method: {self._pixel_qkv_method}")

        self._total_key_depth = total_key_depth
        self._total_value_depth = total_value_depth
        self._num_heads = num_heads
        self._bn_layer = norm_layer
        self._conv_kernel_weight_decay = conv_kernel_weight_decay
        self._position_embedding_method = position_embedding_method
        self._position_embedding_stride = position_embedding_stride
        self._sp_position_embedding_stride = (
            sp_position_embedding_stride or position_embedding_stride
        )

        if self._position_embedding_method == "none":
            pass
        elif self._position_embedding_method == "learnable":
            self.sp_pos_embed = self._create_pos_embed(
                in_features, superpixel_shape, self._sp_position_embedding_stride
            )
            self.pixel_pos_embed = self._create_pos_embed(
                in_features, pixel_shape, self._position_embedding_stride
            )
            print(
                f"sp_pos_embed shape: {self.sp_pos_embed.shape}, "
                f"pixel_pos_embed shape: {self.pixel_pos_embed.shape}"
            )
        elif self._position_embedding_method == "depthwise":
            conv_fn = partial(
                tml.create_conv2d,
                in_features,
                in_features,
                3,
                bias=True,
                depthwise=True,
            )
            self.sp_pos_conv = conv_fn()
            self.pixel_pos_conv = conv_fn()
        else:
            raise ValueError()

        if ls_init_value is not None:
            self.sp_ls1 = LayerScale2d(in_features, ls_init_value)
            if pixelify:
                self.pixel_ls1 = LayerScale2d(in_features, ls_init_value)
            else:
                self.pixel_ls1 = nn.Identity()
        else:
            self.sp_ls1 = nn.Identity()
            self.pixel_ls1 = nn.Identity()

        if temperature_method == "none":
            self.temperature_sp = self.temperature_pixel = 1.0
        elif temperature_method == "shared":
            self.temperature_sp = self.temperature_pixel = nn.Parameter(torch.ones(1))
        elif temperature_method == "individual":
            self.temperature_sp = nn.Parameter(torch.ones(1))
            # NOTE(meijieru): if not pixelify, no gradient to update it
            self.temperature_pixel = nn.Parameter(torch.ones(1)) if self._pixelify else 1.0
        else:
            raise ValueError(f"Unknown learnable_scale_method: {temperature_method}")

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
        self, val: torch.Tensor, raw_val: Optional[torch.Tensor], qkv_method: str
    ) -> Sequence[torch.Tensor]:
        if qkv_method == "linear":
            query, key, value = torch.split(
                val,
                [
                    self._total_key_depth,
                    self._total_key_depth,
                    self._total_value_depth,
                ],
                dim=1,
            )
            query, key, value = [
                reshape_and_transpose_for_attention_operation(val, self._num_heads)
                for val in [query, key, value]
            ]
        elif qkv_method in ["linear_qk", "linear_qk_rawv"]:
            query, key = torch.split(
                val,
                [
                    self._total_key_depth,
                    self._total_key_depth,
                ],
                dim=1,
            )
            value = raw_val
            query, key, value = [
                reshape_and_transpose_for_attention_operation(val, self._num_heads)
                for val in [query, key, value]
            ]

        elif qkv_method in ["identity", "linear_shared", "norm"]:
            query = key = value = reshape_and_transpose_for_attention_operation(
                val, self._num_heads
            )
        else:
            raise ValueError(f"Unknown pixel_qkv_method: {self._pixel_qkv_method}")

        return query, key, value

    def _expand_single_sp_features(self, val: torch.Tensor, b: int):
        # TODO(meijieru): temporarily solution to deal with the heads.
        assert val.shape[0] == 1
        return val.expand(b, -1, -1, -1, -1)

    def _superpixel_update_step(
        self,
        sp_query: torch.Tensor,
        pixel_key: torch.Tensor,
        pixel_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsp, _, _, sh, sw = sp_query.shape
        b, num_heads, c_key, h, w = pixel_key.shape
        _, _, c_value, _, _ = pixel_value.shape
        if bsp == 1:
            sp_query = self._expand_single_sp_features(sp_query, b)
        else:
            assert b == bsp
        scale = c_key**-0.5
        # [b * num_heads]
        sp_query, pixel_key, pixel_value = [
            val.flatten(end_dim=1) for val in [sp_query, pixel_key, pixel_value]
        ]
        similarities = (
            superpixel_ops.compute_similarities_dot_product(pixel_key, sp_query)
            * scale
            * self.temperature_sp
        )
        sp_delta = superpixel_ops.update_superpixel_features(
            pixel_value, sp_query, similarities
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

        bsp, _, _, _, _ = sp_key.shape
        b, num_heads, c_key, h, w = pixel_query.shape
        _, _, c_value, _, _ = sp_value.shape
        if bsp == 1:
            sp_key = self._expand_single_sp_features(sp_key, b)
            sp_value = self._expand_single_sp_features(sp_value, b)
        else:
            assert b == bsp
        scale = c_key**-0.5
        pixel_query, sp_key, sp_value = [
            val.flatten(end_dim=1) for val in [pixel_query, sp_key, sp_value]
        ]
        similarities = (
            superpixel_ops.compute_similarities_dot_product(pixel_query, sp_key)
            * scale
            * self.temperature_pixel
        )
        pixel_delta = None
        if self._pixelify:
            pixel_delta = superpixel_ops.update_pixel_features(
                None, sp_value, similarities
            )
            pixel_delta = pixel_delta.reshape(b, num_heads * c_value, h, w)
        return pixel_delta, similarities.reshape(b, num_heads, 9, h, w)

    def forward(self, inputs):
        """Performs a forward pass."""
        # [b, c, h, w], [b, csp, sh, sw]
        pixel_features, sp_features = inputs

        sp_features_embeded = sp_features
        pixel_features_embeded = pixel_features
        if self._position_embedding_method == "learnable":
            sp_features_embeded = self._add_pos_embed(
                sp_features_embeded, self.sp_pos_embed
            )
            pixel_features_embeded = self._add_pos_embed(
                pixel_features_embeded, self.pixel_pos_embed
            )
        elif self._position_embedding_method == "depthwise":
            sp_features_embeded = (
                self.sp_pos_conv(sp_features_embeded) + sp_features_embeded
            )
            pixel_features_embeded = (
                self.pixel_pos_conv(pixel_features_embeded) + pixel_features_embeded
            )

        sp_query, sp_key, sp_value = self._split_qkv(
            self._sp_qkv(sp_features_embeded), None, "linear"
        )

        pixel_value = None
        if self._pixel_qkv_method == "linear_qk_rawv":
            pixel_value = pixel_features
        elif self._pixel_qkv_method == "linear_qk":
            pixel_value = pixel_features_embeded
        pixel_query, pixel_key, pixel_value = self._split_qkv(
            self._pixel_qkv(pixel_features_embeded), pixel_value, self._pixel_qkv_method
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
            pixel_return = pixel_features + self.pixel_ls1(pixel_features_delta)
        else:
            pixel_return = None

        return (
            pixel_return,
            sp_features + self.sp_ls1(sp_feature_delta),
            similarities_multi_head,
            similarities_multi_head_pixel,
        )

    # def init_weights(self):
    #     # initialization_std = self._bottleneck_channels**-0.5
    #     initialization_std = 0.02
    #     for conv in [
    #         self._memory_qkv_conv_bn.conv,
    #         self._pixel_qkv_conv_bn.conv,
    #     ]:
    #         nn.init.trunc_normal_(conv.weight, std=initialization_std)
    #         if conv.bias is not None:
    #             nn.init.zeros_(conv.bias)

    #     # for bn in [
    #     #     self._memory_conv3_bn.bn,
    #     #     # self._memory_ffn[-1].bn,
    #     #     self._pixel_conv3_bn.bn,
    #     # ]:
    #     #     nn.init.constant_(bn.weight, 1e-3)
    #     #     nn.init.zeros_(bn.bias)   #     #     nn.init.zeros_(bn.bias)