"""SuperPixel Transformer heads."""

from functools import partial
from typing import Any, Callable, MutableMapping, Optional, Sequence, Tuple

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.layers as timm_layers
from timm.models.layers import create_conv2d
from timm.models.convnext import ConvNeXtBlock

from . import superpixel_ops as ops
from . import dual_path_transformer_ops as dual_path
from . import superpixel_cross_attention_ops as sp_cross_ops
from . import superpixel_cross_attention_new_ops as sp_cross_new_ops

rearrange = einops.rearrange

reshape_and_transpose_for_attention_operation = (
    sp_cross_ops.reshape_and_transpose_for_attention_operation
)
LayerScale2d = sp_cross_ops.LayerScale2d


def get_final_similarity(
    info: Optional[MutableMapping[str, torch.Tensor]],
    index: int,
    pixel: bool = False,
    merge: bool = True,
) -> Optional[torch.Tensor]:
    if info is None:
        return None

    if "similarities_final" in info:
        if not merge:
            raise NotImplementedError("Not implemented")
        return info["similarities_final"]

    key = f"similarities_pixel_{index}" if pixel else f"similarities_sp_{index}"
    if key in info:
        similarities = info[key]
    else:
        # For compatibility with old checkpoints.
        # raise ValueError(f"{info.keys()}, similarities_{index} deprecated")
        similarities = info[f"similarities_{index}"]

    if merge and similarities.dim() == 5:
        # For compatibility with old checkpoints.

        # NOTE(meijieru): $simil = qk / \sqrt{c}$, to convert it into
        # single head similarities, so we need to multiply the normalizer
        _, num_heads, _, _, _ = similarities.shape
        similarities = similarities.sum(1) / math.sqrt(num_heads)

    return similarities


class HyperColumn(nn.Module):
    """Fuses multiscale features."""

    def __init__(
        self,
        input_channels_list: Sequence[int],
        output_shape: Tuple[int, int],
        output_channels: Optional[int],
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels_list = input_channels_list
        self.output_shape = output_shape
        self.output_channels = output_channels
        self.align_corners = align_corners

        self.linears = nn.ModuleList(
            [
                # NOTE(meijieru): we don't have normalization & activation here.
                nn.Conv2d(input_channels, self.output_channels, 1)
                if output_channels is not None
                else nn.Identity()
                for input_channels in input_channels_list
            ]
        )

    def resize_on_demand(self, val: torch.Tensor, **kwargs):
        _, _, ih, iw = val.shape
        if ih == self.output_shape[0] and iw == self.output_shape[1]:
            return val
        else:
            return F.interpolate(
                val,
                size=self.output_shape,
                mode="bilinear",
                align_corners=self.align_corners,
                **kwargs,
            )

    def forward(self, features_list: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(features_list) == len(self.linears)
        outputs = [
            self.resize_on_demand(module(val))
            for module, val in zip(self.linears, features_list)
        ]
        return sum(outputs)  # type: ignore[misc]


class ConvStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels_list: Sequence[int] = (48, 96, 192, 384, 384),
        kernel_sizes: Sequence[int] = (3, 3, 3, 3, 1),
        strides: Sequence[int] = (2, 2, 2, 2, 1),
        conv_types: Sequence[str] = ("conv", "conv", "conv", "conv", "conv"),
        norm_layer=nn.BatchNorm2d,
        act_layer=partial(nn.ReLU, inplace=True),
    ) -> None:
        super().__init__()
        assert (
            
            len(channels_list) == len(strides) == len(kernel_sizes) == len(conv_types)
        )

        previous_channels = in_channels
        self.conv_layers = nn.ModuleList()
        for kernel_size, stride, output_channels, conv_type in zip(
            kernel_sizes, strides, channels_list, conv_types
        ):
            if conv_type == "conv":
                layer = nn.Sequential(
                    *[
                        create_conv2d(
                            in_channels=previous_channels,
                            out_channels=output_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding="same",
                            bias=False,
                        ),
                        norm_layer(output_channels),
                        act_layer(),
                    ]
                )
            elif conv_type == "patch":
                layer = nn.Sequential(
                    *[
                        create_conv2d(
                            in_channels=previous_channels,
                            out_channels=output_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding="same",
                            bias=False,
                        ),
                        norm_layer(output_channels),
                    ]
                )
            elif conv_type == "lift_avgpool":
                layer = nn.Sequential(
                    timm_layers.create_conv2d(
                        previous_channels, output_channels, 1, padding="same"
                    ),
                    norm_layer(output_channels),
                    act_layer(),
                    nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size),
                )
            elif conv_type == "avgpool_lift":
                layer = nn.Sequential(
                    nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size),
                    timm_layers.create_conv2d(
                        previous_channels, output_channels, 1, padding="same"
                    ),
                    norm_layer(output_channels),
                    act_layer(),
                )
            elif conv_type == "convnext":
                layer = ConvNeXtBlock(
                    previous_channels,
                    output_channels,
                    kernel_size,
                    stride=stride,
                    mlp_ratio=1,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            elif conv_type == "sep_conv":
                layer = nn.Sequential(
                    timm_layers.separable_conv.SeparableConv2d(
                        previous_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding="same",
                        bias=False,
                    ),
                    norm_layer(output_channels),
                    act_layer(),
                )
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            self.conv_layers.append(layer)
            previous_channels = output_channels

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[MutableMapping[str, torch.Tensor], torch.Tensor]:
        res = {}
        for i, layer in enumerate(self.conv_layers):
            res[f"conv{i}"] = x = layer(x)
        return res, x


class SimilaritiesHead(nn.Module):
    """3h * 3w pixels as channels."""

    def __init__(
        self,
        pixel_shape,
        superpixel_shape,
        embed_dim: int,
        num_classes: int,
        kernel_size: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pixel_shape = pixel_shape
        self.superpixel_shape = superpixel_shape
        self.embed_dim = embed_dim

        assert len(self.pixel_shape) == len(self.superpixel_shape)
        assert superpixel_shape[0] == superpixel_shape[1]
        strides = {7: [1, 1, 1], 14: [2, 1, 1]}[superpixel_shape[1]]
        sh, sw = self.superpixel_shape
        h, w = self.pixel_shape
        self.convs = ConvStem(
            9 * h * w // sh // sw,
            [embed_dim for _ in strides],
            [kernel_size for _ in strides],
            strides,
            **kwargs,
        )
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        similarities_sp_persptive = ops.softmax_along_superpixel(similarities)
        similarities_sp_persptive = similarities_sp_persptive.flatten(1, 3)
        _, feat = self.convs(similarities_sp_persptive)
        feat = feat.mean(3).mean(2)
        feat = self.fc_norm(feat)
        return self.head(feat)


class SuperPixelStem(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans=3,
        hypercolumn_indices: Sequence[int] = (1, 2, 3),
        hypercolumn_dim: Optional[int] = 384,
        stem_channels_list: Sequence[int] = (48, 96, 192, 384),
        strides: Sequence[int] = (2, 2, 2, 2),
        pixel_stride: int = 4,
        superpixel_iter: int = 1,
        superpixel_features_init_method: str = "avgpool",
        position_embedding_method: str = "none",
        position_embedding_stride: int = -1,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        pre_norm_layer: Optional[Callable] = None,
        proj_norm_layer: Optional[Callable] = None,
        tokenization_method: str = "identity",
        **tokenization_kwargs,
    ) -> None:
        super().__init__()

        assert img_size % pixel_stride == 0 and img_size % patch_size == 0
        assert len(stem_channels_list) == 4
        assert hypercolumn_indices[-1] == len(stem_channels_list) - 1

        self.conv_stem = ConvStem(
            in_channels=in_chans,
            channels_list=stem_channels_list,
            kernel_sizes=[3 for _ in stem_channels_list],
            strides=strides,
        )

        self.pixel_stride = pixel_stride
        self.pixel_output_shape = tuple([img_size // pixel_stride for _ in range(2)])
        self.hypercolumn_indices = hypercolumn_indices
        self.hyper_column = HyperColumn(
            [stem_channels_list[i] for i in hypercolumn_indices],
            output_shape=self.pixel_output_shape,
            output_channels=hypercolumn_dim,
            align_corners=False,
        )
        if hypercolumn_dim is None:
            hypercolumn_dim = stem_channels_list[-1]

        self.tokenization_method = tokenization_method
        self.superpixel_shape = tuple([img_size // patch_size for _ in range(2)])
        if self.tokenization_method == "identity":
            # NOTE(meijieru): don't honor num_heads_cross_attention
            self.superpixel_tokenize = SuperPixelTokenizationIdentity(
                superpixel_iter,
                1,
                hypercolumn_dim,
                self.pixel_output_shape,
                self.superpixel_shape,
                superpixel_features_init_method=superpixel_features_init_method,
                pixel_features_update_method="fixed",
                superpixel_features_update_method="residual_only",
                **tokenization_kwargs,
            )
        elif self.tokenization_method == "dual_path":
            self.superpixel_tokenize = SuperPixelTokenizationDualPath(
                superpixel_iter,
                1,
                hypercolumn_dim,
                self.pixel_output_shape,
                self.superpixel_shape,
                superpixel_features_init_method=superpixel_features_init_method,
                position_embedding_method=position_embedding_method,
                position_embedding_stride=position_embedding_stride,
                feed_forward_network_channels=-1,
                **tokenization_kwargs,
            )
        elif self.tokenization_method == "sp_cross":
            self.superpixel_tokenize = SuperPixelTokenizationCrossAttention(
                superpixel_iter,
                1,
                hypercolumn_dim,
                self.pixel_output_shape,
                self.superpixel_shape,
                superpixel_features_init_method=superpixel_features_init_method,
                position_embedding_method=position_embedding_method,
                position_embedding_stride=position_embedding_stride,
                **tokenization_kwargs,
            )
        elif self.tokenization_method == "patch":
            sp_size = patch_size // pixel_stride
            self.avgpool = nn.AvgPool2d(kernel_size=sp_size, stride=sp_size)
        else:
            raise ValueError()

        def apply_norm_layer(layer):
            if layer is None:
                return nn.Identity()

            layer_func = layer.func if isinstance(layer, partial) else layer.__class__
            if layer_func is sp_cross_ops.LayerNorm2DChannelOnly:
                return layer(hypercolumn_dim)
            elif layer_func is nn.LayerNorm:
                return layer((hypercolumn_dim, *self.pixel_output_shape))
            else:
                raise ValueError(f"Unknown type: {layer_func}")

        self.pre_norm = apply_norm_layer(pre_norm_layer)
        self.proj = nn.Conv2d(hypercolumn_dim, embed_dim, 1, bias=bias)
        # TODO(meijieru): deprecated
        self.norm = apply_norm_layer(proj_norm_layer)
        self.flatten = flatten
        self.num_patches = math.prod(self.superpixel_shape)

    @property
    def tokenization_info(self) -> Optional[MutableMapping[str, torch.Tensor]]:
        return getattr(self, "_tokenization_info", None)

    @property
    def similarities_final(self) -> Optional[torch.Tensor]:
        return get_final_similarity(
            self.tokenization_info, self.superpixel_tokenize.num_blocks
        )

    def forward(self, x: torch.Tensor):
        endpoints, _ = self.conv_stem(x)
        pixel_features = self.hyper_column(
            [endpoints[f"conv{i}"] for i in self.hypercolumn_indices]
        )
        pixel_features = self.pre_norm(pixel_features)

        if self.tokenization_method == "patch":
            sp_features = self.avgpool(pixel_features)
        else:
            tokenization_info, _, sp_features = self.superpixel_tokenize(pixel_features)
            self._tokenization_info = {
                key: val
                for key, val in tokenization_info.items()
                if "similarities" in key
            }
            index = self.superpixel_tokenize.num_blocks
            assert sp_features is tokenization_info[f"superpixel_features_{index}"]

        sp_features = self.proj(sp_features)
        sp_features = self.norm(sp_features)
        if self.flatten:
            sp_features = sp_features.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return sp_features


class SuperPixelTokenization(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        num_channels: int,
        pixel_shape: Tuple[int, int],
        superpixel_shape: Tuple[int, int],
        superpixel_features_update_method: str = "residual",
        pixel_features_update_method: str = "residual",
        superpixel_features_init_method: str = "learnable",
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.superpixel_features_init_method = superpixel_features_init_method
        self.pixel_shape = pixel_shape
        self.superpixel_shape = superpixel_shape
        self.superpixel_features_update_method = superpixel_features_update_method
        self.pixel_features_update_method = pixel_features_update_method

        # TODO(meijieru): revisit initialization.
        if self.superpixel_features_init_method == "learnable":
            self.init_superpixel_queries = nn.Parameter(
                torch.randn(1, num_channels, *superpixel_shape) * 0.02
            )
        elif self.superpixel_features_init_method in ["avgpool", "from_feature"]:
            pass
        else:
            raise ValueError()
        self.similarity_scale = (self.num_channels // self.num_heads) ** -0.5

    def _compute_init_superpixel_queries(
        self,
        init_pixel_features: torch.Tensor,
        sp_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.superpixel_features_init_method == "from_feature":
            assert sp_features is not None
            return sp_features

        elif self.superpixel_features_init_method == "learnable":
            init_superpixel_queries = self.init_superpixel_queries
        elif self.superpixel_features_init_method == "avgpool":
            sh, sw = self.superpixel_shape
            _, _, h, w = init_pixel_features.shape
            init_superpixel_queries = F.avg_pool2d(
                init_pixel_features, (h // sh, w // sw)
            )
        else:
            raise ValueError()
        return init_superpixel_queries


def prepare_similarities(
    patch_embed: SuperPixelStem,
    similarities: torch.Tensor,
    resize: bool = True,
    stride: int = 2,
) -> torch.Tensor:
    b, _, h, w = similarities.shape
    sh, sw = patch_embed.superpixel_shape
    ph, pw = h // sh, w // sw

    # generate seg at stride 2
    scale_factor = patch_embed.pixel_stride // stride
    similarities = similarities.reshape(b, 9, sh, ph, sw, pw)
    if resize:
        similarities = ops.resize_similarities(
            similarities,
            scale_factor=scale_factor,
        )
        similarities = ops.maskout_boundary(similarities)
    return similarities


def visualize_superpixel(
    tokenization_info, patch_embed, resize_similarities: bool = True
):
    assert tokenization_info is not None
    colormap = ops.create_superpixel_colormap("random")

    res = {}
    for key, similarities in tokenization_info.items():
        similarities = prepare_similarities(
            patch_embed, similarities, resize=resize_similarities
        )
        labels = ops.compute_hard_association(similarities).cpu()
        vis = colormap[labels % colormap.size(0)]
        res[key] = vis
    return res


class SuperPixelTokenizationIdentity(SuperPixelTokenization):
    """
    NOTE(meijieru): doesn't honor num_heads.
    """

    def forward(
        self,
        init_pixel_features: torch.Tensor,
        sp_features: Optional[torch.Tensor] = None,
    ) -> Tuple[MutableMapping[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        init_superpixel_queries = self._compute_init_superpixel_queries(
            init_pixel_features, sp_features=sp_features
        )
        init_similarities = (
            ops.compute_similarities_dot_product(
                init_pixel_features, init_superpixel_queries
            )
            * self.similarity_scale
        )
        cur = (init_pixel_features, init_superpixel_queries, init_similarities)
        res = {
            "pixel_features_0": cur[0],
            "superpixel_features_0": cur[1],
            "similarities_0": cur[2],
        }
        for i in range(1, self.num_blocks + 1):
            pixel_features = self._update_pixel_features(*cur)
            sp_features = self._update_superpixel_features(*cur)
            similarities = (
                ops.compute_similarities_dot_product(pixel_features, sp_features)
                * self.similarity_scale
            )

            cur = (pixel_features, sp_features, similarities)
            res.update(
                {
                    f"pixel_features_{i}": cur[0],
                    f"superpixel_features_{i}": cur[1],
                    f"similarities_{i}": cur[2],
                }
            )

        return res, pixel_features, sp_features

    def _update_superpixel_features(
        self,
        pixel_features: torch.Tensor,
        sp_features: torch.Tensor,
        similarities: torch.Tensor,
    ) -> torch.Tensor:
        sp_delta = ops.update_superpixel_features(
            pixel_features, sp_features, similarities
        )
        if self.superpixel_features_update_method == "residual":
            return sp_delta + sp_features
        elif self.superpixel_features_update_method == "residual_only":
            return sp_delta
        else:
            raise ValueError()

    def _update_pixel_features(
        self,
        pixel_features: torch.Tensor,
        sp_features: torch.Tensor,
        similarities: torch.Tensor,
    ) -> torch.Tensor:
        if self.pixel_features_update_method == "fixed":
            return pixel_features

        pixel_delta = ops.update_pixel_features(
            pixel_features, sp_features, similarities
        )
        if self.pixel_features_update_method == "residual":
            return pixel_delta + pixel_features
        elif self.pixel_features_update_method == "residual_only":
            return pixel_delta
        else:
            raise ValueError()


class SuperPixelTokenizationCrossAttention(SuperPixelTokenization):
    def __init__(
        self,
        *args,
        position_embedding_method: str = "none",
        position_embedding_stride=-1,
        sp_position_embedding_stride=None,
        return_similarities_final: bool = True,
        return_final_pixel_features: bool = False,
        key_expansion=1,
        value_expansion=1,
        pixel_qkv_method: str = "linear",
        layer_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert self.superpixel_features_update_method == "residual"
        assert self.pixel_features_update_method in ["residual", "fixed"]
        print(f"pixel_features_update_method is {self.pixel_features_update_method}")
        self.return_similarities_final = return_similarities_final

        layer_kwargs = layer_kwargs or {}
        if (
            return_similarities_final
            and layer_kwargs.get("learnable_scale_method", "none") != "none"
        ):
            raise NotImplementedError(
                "learnable_scale_method is not supported when return_similarities_final is True"
            )

        blocks = [
            sp_cross_ops.DualPathTransformerSimpleLayer(
                self.num_channels,
                filters=self.num_channels,
                num_heads=self.num_heads,
                pixel_shape=self.pixel_shape,
                superpixel_shape=self.superpixel_shape,
                position_embedding_method=position_embedding_method,
                position_embedding_stride=position_embedding_stride,
                sp_position_embedding_stride=sp_position_embedding_stride,
                key_expansion=key_expansion,
                value_expansion=value_expansion,
                pixel_qkv_method=pixel_qkv_method,
                pixelify=(
                    self.return_similarities_final
                    or i != self.num_blocks - 1
                    or return_final_pixel_features
                )
                and self.pixel_features_update_method == "residual",
                **layer_kwargs,
            )
            for i in range(self.num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, pixel_features: torch.Tensor, sp_features: Optional[torch.Tensor] = None
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        sp_features = self._compute_init_superpixel_queries(
            pixel_features, sp_features=sp_features
        )
        res = {"pixel_features_0": pixel_features, "superpixel_features_0": sp_features}
        for i, block in enumerate(self.blocks, start=1):
            (
                pixel_features_updated,
                sp_features,
                similarities_multi_head_sp,
                similarities_multi_head_pixel,
            ) = block((pixel_features, sp_features))
            if self.pixel_features_update_method == "fixed":
                del pixel_features_updated
            elif self.pixel_features_update_method == "residual":
                pixel_features = pixel_features_updated
            else:
                raise ValueError()
            assert (
                similarities_multi_head_sp.dim() == 5
                and similarities_multi_head_sp.size(1) == self.num_heads
            )
            assert similarities_multi_head_pixel is None or (
                similarities_multi_head_pixel.dim() == 5
                and similarities_multi_head_pixel.size(1) == self.num_heads
            )
            res_iter = {
                f"pixel_features_{i}": pixel_features,
                f"superpixel_features_{i}": sp_features,
                f"similarities_sp_{i}": similarities_multi_head_sp,
            }
            if similarities_multi_head_pixel is not None:
                res_iter[f"similarities_pixel_{i}"] = similarities_multi_head_pixel
            res.update(res_iter)
        if self.return_similarities_final:
            res["similarities_final"] = (
                ops.compute_similarities_dot_product(pixel_features, sp_features)
                * self.similarity_scale
            )

            if False:
                norm_layer = sp_cross_ops.LayerNorm2DChannelOnly(
                    self.num_channels, elementwise_affine=False
                )
                res["similarities_final_normalized"] = (
                    ops.compute_similarities_dot_product(
                        norm_layer(pixel_features), norm_layer(sp_features)
                    )
                    * self.similarity_scale
                )

        return res, pixel_features, sp_features


class SuperPixelTokenizationCrossAttentionAsymmetry(SuperPixelTokenization):
    def __init__(
        self,
        *args,
        num_channels_sp: int = -1,
        num_heads_sp: int = -1,
        position_embedding_method: str = "none",
        position_embedding_stride=-1,
        sp_position_embedding_stride=None,
        return_similarities_final: bool = True,
        return_final_pixel_features: bool = False,
        pixel_qkv_method: str = "linear",
        layer_kwargs=None,
        reweight_sp: bool = False,
        reweight_pixel: bool = False,
        
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_channels_sp = num_channels_sp
        self.num_heads_sp = num_heads_sp

        assert self.superpixel_features_init_method == "from_feature"
        assert self.superpixel_features_update_method in ["residual"]
        assert self.pixel_features_update_method in ["residual", "fixed"]
        print(
            f"pixel_features_update_method: {self.pixel_features_update_method}, "
            f"superpixel_features_update_method: {self.superpixel_features_update_method}"
        )
        self.return_similarities_final = return_similarities_final

        layer_kwargs = layer_kwargs or {}
        blocks = [
            sp_cross_new_ops.DualPathCrossAttentionLayer(
                self.num_channels,
                self.num_channels_sp,
                (self.num_channels, self.num_channels, self.num_channels_sp),
                (self.num_channels, self.num_channels, self.num_channels),
                num_heads_pixel=self.num_heads,
                num_heads_sp=self.num_heads_sp,
                pixel_shape=self.pixel_shape,
                superpixel_shape=self.superpixel_shape,
                position_embedding_method=position_embedding_method,
                position_embedding_stride=position_embedding_stride,
                sp_position_embedding_stride=sp_position_embedding_stride,
                pixel_qkv_method=pixel_qkv_method,
                pixelify=(
                    self.return_similarities_final
                    or i != self.num_blocks - 1
                    or return_final_pixel_features
                )
                and self.pixel_features_update_method == "residual",
                # hard_assign=True if i == self.num_blocks - 1 else False,
                hard_assign = False,
                reweight_sp = reweight_sp,
                reweight_pixel = reweight_pixel, 
                **layer_kwargs,
            )
            for i in range(self.num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, pixel_features: torch.Tensor, sp_features: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        sp_features = self._compute_init_superpixel_queries(
            pixel_features, sp_features=sp_features
        )
        res = {"pixel_features_0": pixel_features, "superpixel_features_0": sp_features}
        for i, block in enumerate(self.blocks, start=1):
            (
                pixel_features_updated,
                sp_features,
                similarities_multi_head_sp,
                similarities_multi_head_pixel,
            ) = block(pixel_features, sp_features)
            if self.pixel_features_update_method == "fixed":
                del pixel_features_updated
            elif self.pixel_features_update_method == "residual":
                pixel_features = pixel_features_updated
            else:
                raise ValueError()
            assert (
                similarities_multi_head_sp.dim() == 5
                and similarities_multi_head_sp.size(1) == self.num_heads
            )
            assert similarities_multi_head_pixel is None or (
                similarities_multi_head_pixel.dim() == 5
                and similarities_multi_head_pixel.size(1) == self.num_heads
            )
            res_iter = {
                f"pixel_features_{i}": pixel_features,
                f"superpixel_features_{i}": sp_features,
                f"similarities_sp_{i}": similarities_multi_head_sp,
            }
            if similarities_multi_head_pixel is not None:
                res_iter[f"similarities_pixel_{i}"] = similarities_multi_head_pixel
            res.update(res_iter)
        if self.return_similarities_final:
            res["similarities_final"] = (
                ops.compute_similarities_dot_product(pixel_features, sp_features)
                * self.similarity_scale
            )

        return res, pixel_features, sp_features


class SuperPixelTokenizationDualPath(SuperPixelTokenization):
    def __init__(
        self,
        *args,
        activation="gelu",
        feed_forward_network_channels: int = 256,
        position_embedding_method: str = "none",
        position_embedding_stride=-1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert self.superpixel_features_update_method == "residual"
        assert self.pixel_features_update_method == "residual"

        blocks = [
            dual_path.DualPathTransformerLayer(
                self.num_channels,
                activation=activation,
                filters=128,
                num_heads=self.num_heads,
                feed_forward_network_channels=feed_forward_network_channels,
                use_memory_self_attention=False,
                use_superpixel_cross_attention=True,
                use_pixel2memory_feedback_attention=(i + 1 == self.num_blocks),
                pixel_shape=self.pixel_shape,
                superpixel_shape=self.superpixel_shape,
                position_embedding_method=position_embedding_method,
                position_embedding_stride=position_embedding_stride,
            )
            for i in range(self.num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, pixel_features: torch.Tensor, sp_features: Optional[torch.Tensor] = None
    ):
        sp_features = self._compute_init_superpixel_queries(
            pixel_features, sp_features=sp_features
        )
        res = {"pixel_features_0": pixel_features, "superpixel_features_0": sp_features}
        for i, block in enumerate(self.blocks, start=1):
            pixel_features, sp_features, similarities_multi_head = block(
                (pixel_features, sp_features)
            )
            assert (
                similarities_multi_head.dim() == 5
                and similarities_multi_head.size(1) == self.num_heads
            )
            res.update(
                {
                    f"pixel_features_{i}": pixel_features,
                    f"superpixel_features_{i}": sp_features,
                    # NOTE(meijieru): $simil = qk / \sqrt{c}$, to convert it into
                    # single head similarities, so we need to multiply the normalizer
                    f"similarities_{i}": similarities_multi_head.sum(1)
                    / math.sqrt(self.num_heads),
                }
            )
        return res, pixel_features, sp_features
