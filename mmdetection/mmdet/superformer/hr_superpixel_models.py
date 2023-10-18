from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence, Tuple, Union
import einops
import warnings
import math
from matplotlib.dates import TU

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm
from timm.models.vision_transformer import Block, _cfg
from timm.models.registry import register_model
from timm.models import layers as timm_layers

from superpixel import superpixel_transformer as st
from superpixel import superpixel_ops


LayerScale2d = st.LayerScale2d
SKIP_CONFIRM = False


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
        pre_norm_pixel: bool = False,
        pixel_refine_method: str = "identity",
        return_updated_pixel_features: bool = False,
        unflatten_sp_features: bool = False,
        use_pos_embed: bool = False,
        use_cls_token: bool = False,
        no_embed_class: bool = False,
        use_pixel_similarities: bool = False,
        use_middle_pixel_features: bool = False,
        merge_multihead_similarities: bool = True,
    ) -> None:
        super().__init__()

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
        self.merge_multihead_similarities = merge_multihead_similarities
        self.downsample = create_downsample(
            stride, in_channels, out_channels, norm_layer_2d, pre_norm_pixel
        )
        self.sp_downsample = create_downsample(
            sp_stride,
            in_channels,
            out_channels,
            norm_layer_2d,
            False,  # FIXME(meijieru): already normed in last stage
        )

        if use_pixel_similarities:
            warnings.warn(
                "`use_pixel_similarities` is for compatible with old models only"
            )
            if not SKIP_CONFIRM:
                __import__("ipdb").set_trace()
        # if not use_pixel_similarities:
        #     warnings.warn(
        #         "not `use_pixel_similarities` is for compatible with old models only"
        #     )
        if merge_multihead_similarities and (
            getattr(self.patch_embed, "num_heads", 1) > 1
        ):
            warnings.warn(
                "`merge_multihead_similarities` is for compatible with old models only"
            )
            if not SKIP_CONFIRM:
                __import__("ipdb").set_trace()

        if (
            isinstance(self.patch_embed, nn.AvgPool2d)
            or self.patch_embed.superpixel_features_init_method != "from_feature"
            or sp_embed_method == "identity"
        ):
            self.sp_transform = nn.Identity()
        elif sp_embed_method == "conv3x3_norm":
            self.sp_transform = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
                norm_layer_2d(out_channels),
            )
        elif sp_embed_method == "conv1x1_norm":
            self.sp_transform = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 1, padding="same"
                ),
                norm_layer_2d(out_channels),
            )
        elif sp_embed_method == "conv3x3_norm_act":
            self.sp_transform = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
                norm_layer_2d(out_channels),
                act_layer(),
            )
        elif sp_embed_method == "conv3x3":
            self.sp_transform = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
            )
        else:
            raise ValueError(f"Unknown sp_embed_method: {sp_embed_method}")

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
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, Sequence)
                        else drop_path_rate
                    ),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    # init_values=ls_init_value,
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
            sp_features_last = self.sp_downsample(sp_features_last)
            sp_features_last = self.sp_transform(sp_features_last)
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
                merge=self.merge_multihead_similarities,
            )
            if similarity is None:
                raise ValueError("Similarity is None")

            if self.merge_multihead_similarities:
                assert similarity.dim() == 4
                pixel_delta = superpixel_ops.update_pixel_features(
                    None, sp_features_unflatten, similarity
                )
            else:
                assert similarity.dim() == 5
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
        res = x + pixel_delta
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
        x = self.downsample(x)# downsample the pixel by param "stides"
        
        info, sp_features, pixel_features_middle = self.forward_patchify(
            x, sp_features_last
        )# sp_feature H//32, W//32, 16,16
        assert sp_features.dim() == 4
        sp_features = sp_features.flatten(2).transpose(1, 2)  # BCHW -> BNC

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
            updated_pixel_features = self.forward_pixelify(
                x if not self.use_middle_pixel_features else pixel_features_middle,
                sp_features_unflatten,
                info,
            )

        if self.unflatten_sp_features:
            if sp_features_unflatten is None:
                sp_features_unflatten = self.forward_unflatten_sp_features(sp_features)
            sp_features = sp_features_unflatten

        return (
            updated_pixel_features,
            sp_features,
            sp_features_seg,
        )


class SuperformerHR(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        in_channels: int = 3,
        num_classes: int = 1000,
        stem_channels_list: Sequence[int] = (32, 32, 64, 64),
        stem_kernel_sizes: Optional[Sequence[int]] = None,
        stem_strides: Sequence[int] = (2, 1, 2, 1),
        stem_conv_types: Sequence[str] = ("conv", "conv", "conv", "conv"),
        hypercolumn_indices: Sequence[int] = (3,),
        hypercolumn_act: bool = True,
        hypercolumn_channels: Optional[int] = None,
        depths: Sequence[int] = (3, 9),
        dims: Sequence[int] = (64, 192),
        heads: Sequence[int] = (1, 3),
        strides: Sequence[int] = (1, 2),
        sp_sizes: Sequence[int] = (4, 4),
        sp_heads: Sequence[int] = (1, 1),
        sp_method: str = "sp_cross",
        sp_iter: int = 2,
        sp_features_init_methods: Sequence[str] = ("avgpool", "avgpool"),
        sp_position_embedding_method: str = "none",
        sp_position_embedding_stride: int = -1,
        sp_sp_position_embedding_stride: Optional[int] = None,
        sp_temperature_method: str = "none",
        sp_pixel_features_update_method: str = "residual",
        sp_kwargs: Optional[Dict[str, Any]] = None,
        conv_norm_layer=None,
        conv_act_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        seg_specific_classifier: str = False,
        seg_num_classes: int = 150,
        num_blocks_after_seg: int = 0,
        ls_init_value: Optional[float] = None,
        sp_ls_init_value: Optional[float] = None,
        pre_norm_pixel: bool = False,
        pixel_refine_method: str = "identity",
        sp_embed_method: str = "identity",
        class_token_position_method: str = "none",
        pos_embed_position_method: str = "none",
        no_embed_class: bool = False,
        fix_drop_path_rate: bool = False,
        weight_init: str = "default",
        use_pixel_similarities: bool = False,
        use_middle_pixel_features: bool = False,
        merge_multihead_similarities: bool = True,
    ):
        super().__init__()
        print(img_size,seg_num_classes)

        if sp_pixel_features_update_method == "fixed" and use_pixel_similarities:
            raise ValueError("pixel_similarities won't be updated")
        
        self.img_size = img_size
        self.sp_method = sp_method
        self.sp_iter = sp_iter
        self.sp_features_init_methods = sp_features_init_methods
        self.sp_position_embedding_method = sp_position_embedding_method
        self.sp_position_embedding_stride = sp_position_embedding_stride
        self.sp_sp_position_embedding_stride = sp_sp_position_embedding_stride
        self.sp_temperature_method = sp_temperature_method
        self.sp_pixel_features_update_method = sp_pixel_features_update_method
        self.sp_ls_init_value = sp_ls_init_value
        self.sp_kwargs = sp_kwargs or {}
        self.use_middle_pixel_features = use_middle_pixel_features

        self.class_token_position_method = class_token_position_method
        self.no_embed_class = no_embed_class
        self.pos_embed_position_method = pos_embed_position_method

        self.seg_specific_classifier = seg_specific_classifier
        self.seg_num_classes = seg_num_classes
        self.num_blocks_after_seg = num_blocks_after_seg

        act_layer = act_layer or nn.GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        norm_layer_2d = partial(get_norm_layer_2d(norm_layer), eps=1e-6)

        conv_norm_layer = conv_norm_layer or partial(timm_layers.LayerNorm2d, eps=1e-6)
        conv_act_layer = conv_act_layer or act_layer

        stem_kernel_sizes = stem_kernel_sizes or [3] * len(stem_channels_list)
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
        pixel_output_shape = tuple([img_size // cur_stride for _ in range(2)])
        hypercolumn_channels = (
            hypercolumn_channels
            if hypercolumn_channels is not None
            else stem_channels_list[-1]
        )
        self.hypercolumn_indices = hypercolumn_indices

        if len(self.hypercolumn_indices) == 1:
            assert self.hypercolumn_indices[0] == len(stem_channels_list) - 1
            self.hyper_column = None
        else:
            self.hyper_column = nn.Sequential(
                *[
                    st.HyperColumn(
                        [stem_channels_list[i] for i in hypercolumn_indices],
                        output_shape=pixel_output_shape,
                        output_channels=hypercolumn_channels,
                        align_corners=False,
                    ),
                    norm_layer_2d(hypercolumn_channels),
                    act_layer() if hypercolumn_act else nn.Identity(),
                ]
            )
        # print(f"depths: {depths}")
        # print(f"dims: {dims}")
        # print(f"heads: {heads}")
        # print(f"sp_sizes: {sp_sizes}")
        # print(f"sp_heads: {sp_heads}")
        # print(f"sp_features_init_methods: {sp_features_init_methods}")

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
        stage_in_dim = hypercolumn_channels
        for i, (depth, dim, head, sp_size, sp_head, stride) in enumerate(
            zip(depths, dims, heads, sp_sizes, sp_heads, strides)
        ):
            cur_stride *= stride
            sp_stride = sp_size * stride // sp_sizes[i - 1] if i > 0 else 1
            sp_layer, sp_shape = self._make_superpixel_layer(
                i, cur_stride, sp_size, sp_head, dim, sp_method
            )

            seg_block_idx = None
            if (
                self.seg_specific_classifier
                and self.num_blocks_after_seg > 0
                and i == num_stages - 1
            ):
                seg_block_idx = depth - self.num_blocks_after_seg

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
                    seg_block_idx=seg_block_idx,
                    ls_init_value=ls_init_value,
                    pre_norm_pixel=pre_norm_pixel_stage,
                    sp_embed_method=sp_embed_method,
                    pixel_refine_method=pixel_refine_method,
                    return_updated_pixel_features=return_updated_pixel_features,
                    unflatten_sp_features=unflatten_sp_features,
                    use_pos_embed=use_pos_embeds[i],
                    use_cls_token=use_class_tokens[i],
                    no_embed_class=self.no_embed_class,
                    superpixel_shape=sp_shape,
                    use_pixel_similarities=use_pixel_similarities,
                    use_middle_pixel_features=use_middle_pixel_features,
                    merge_multihead_similarities=merge_multihead_similarities,
                )
            )
            if fix_drop_path_rate:
                cur_depth += depth

        if fix_drop_path_rate:
            assert cur_depth == sum(depths)
        else:
            warnings.warn("drop_path_rate is not fixed, this is for compatible only")
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
            assert num_blocks_after_seg >= 0
            # NOTE(meijieru): the token features doesn't use avgpool, so we skip
            # the fc_norm.
            if self.seg_specific_classifier == "Linear":
                self.seg_head = nn.Linear(self.embed_dim, self.seg_num_classes)
            elif self.seg_specific_classifier == "Conv":
                self.seg_head_conv = nn.Conv2d(self.embed_dim, self.seg_num_classes,1)
            else:
                print(self.seg_specific_classifier)
                raise ValueError()
            self.seg_norm = norm_layer(self.embed_dim)
            print(f"use {self.num_blocks_after_seg} for fusion seg")

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in (
            "jax",
            "jax_nlhb",
            "moco",
            "",
            "default",
        )
        if mode == "default":
            init_fn = init_superformer_weights
        else:
            raise NotImplementedError(f"Unknown init mode: {mode}")

        timm.models.helpers.named_apply(init_fn, self)

    def _make_superpixel_layer(
        self,
        i: int,
        pixel_stride: int,
        sp_size: int,
        sp_head: int,
        dim: int,
        method: str = "sp_cross",
    ) -> Tuple[Callable, Sequence[int]]:
        assert self.img_size % pixel_stride == 0
        pixel_shape = [self.img_size // pixel_stride for _ in range(2)]
        sp_shape = [val // sp_size for val in pixel_shape]

        if method == "sp_cross":
            if (
                i != len(self.sp_features_init_methods) - 1
                and self.use_middle_pixel_features
            ):
                return_final_pixel_features = True
            else:
                return_final_pixel_features = False
            sp_fn = partial(
                st.SuperPixelTokenizationCrossAttention,
                self.sp_iter,
                sp_head,
                dim,
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
                    "temperature_method": self.sp_temperature_method,
                },
                # TODO(meijieru): pass norm_layer_2d
                **self.sp_kwargs,
            )
        elif method == "identity":
            sp_fn = partial(
                st.SuperPixelTokenizationIdentity,
                self.sp_iter,
                1,
                dim,
                pixel_shape,
                sp_shape,
                superpixel_features_init_method=self.sp_features_init_methods[i],
                pixel_features_update_method="fixed",
                superpixel_features_update_method="residual_only",
                **self.sp_kwargs,
            )
        elif method == "patch":
            sp_fn = partial(nn.AvgPool2d, kernel_size=sp_size, stride=sp_size)
        else:
            raise ValueError(f"Unknown superpixel method {method}")
        return sp_fn, sp_shape

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, MutableMapping[str, torch.Tensor]]:
        endpoints, pixel_features = self.stem(x) #1.过一个stem 下采样到原来的1/4
        if self.hyper_column is not None:
            pixel_features = self.hyper_column(
                [endpoints[f"conv{i}"] for i in self.hypercolumn_indices]
            )
            endpoints.update({"hypercolumn": pixel_features})

        sp_features_last = None
        assert len(self.stages) > 0
        for _, stage in enumerate(self.stages):#2.迭代次数是depth,进行sp和pixel的循环更新
            pixel_features, sp_features, sp_features_seg = stage(
                pixel_features,
                sp_features_last,
            )
            sp_features_last = sp_features

            # # rank not consistent due to whether flatten or not
            # # res[f"sp_features_stage{i}"] = sp_features
            # if return_updated_pixel_features:
            #     endpoints[f"pixel_features_stage{i}"] = pixel_features
        return sp_features, sp_features_seg, endpoints

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
        self, x: torch.Tensor, return_pixel_logits: bool = True, stride: int = 2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.seg_specific_classifier:
            raise ValueError("No segmentation head is found.")
        elif self.seg_specific_classifier=="Linear":
            x = self.seg_norm(x)
            b, num, c = x.shape
            x = x.reshape(b * num, c)

            sp_logits = self.seg_head(x)

            last_stage = self.stages[-1]
            last_sp_layer = last_stage.patch_embed

            if isinstance(last_sp_layer, nn.AvgPool2d):
                sh=sw = last_sp_layer.kernel_size
            elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                sh, sw = last_sp_layer.superpixel_shape
            else:
                raise ValueError()
            sp_logits = sp_logits.view(b, sh, sw, -1).permute(0, 3, 1, 2)
        elif self.seg_specific_classifier=="Conv":
            
            last_stage = self.stages[-1]
            last_sp_layer = last_stage.patch_embed

            if isinstance(last_sp_layer, nn.AvgPool2d):
                sh=sw = last_sp_layer.kernel_size
            elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                sh, sw = last_sp_layer.superpixel_shape
            else:
                raise ValueError()
            x = self.seg_norm(x)
            b, num, c = x.shape

            x = x.view(b,sh,sw,-1).permute(0, 3, 1, 2) #B,C,sh,sw
            sp_logits = self.seg_head_conv(x)

            

        pixel_logits = None
        if return_pixel_logits:
            if isinstance(last_sp_layer, nn.AvgPool2d):
                raise NotImplementedError()
            elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                info = last_stage.tokenization_info
                if info is None:
                    raise ValueError()
                scale_factor = self.img_size // last_sp_layer.pixel_shape[0] // stride

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
        return_pixel_logits: bool = False,
        seg_stride: int = 2,
    ) -> Union[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        sp_features, sp_features_seg, _ = self.forward_features(x)
        logits = self.forward_head(sp_features)
        ret = {"cls": logits}
        if generate_seg:
            sp_logits, pixel_logits = self.forward_segmentation(
                sp_features_seg, return_pixel_logits, stride=seg_stride
            )
            ret["sp_cls"] = sp_logits
            if pixel_logits is not None:
                ret["seg"] = pixel_logits
        return ret

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

    def visualize_superpixel(self, resize_similarities: bool = True):
        res = {}
        for i, stage in enumerate(self.stages):
            patch_embed = stage.patch_embed
            tokenization_info = stage.tokenization_info
            if resize_similarities:
                scale_factor = self.img_size // stage.patch_embed.pixel_shape[0]
            else:
                scale_factor = 1
            if tokenization_info is None:
                continue
            res_i = visualize_superpixel(tokenization_info, patch_embed, scale_factor)
            res.update({f"stage{i}_{key}": val for key, val in res_i.items()})
        return res

@register_model
def hr_superformer_stemp8_patch7_7_base_224_ls_1e_5(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(8,),
        stem_channels_list=(768,),
        stem_conv_types=("patch",),
        stem_kernel_sizes=(8,),
        dims=(768,),
        heads=(12, 12),
        strides=(1, 1),
        hypercolumn_indices=(0,),  # disable hypercolumn
        sp_method="patch",
        ls_init_value=1e-5,
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))

@register_model
def hr_superformer_stemp8_patch7_7_small_224_ls_1e_5(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)

    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(8,),
        stem_channels_list=(384,),
        stem_conv_types=("patch",),
        stem_kernel_sizes=(8,),
        dims=(384, 384),
        # heads=(3, 3),
        strides=(1, 1),
        hypercolumn_indices=(0,),  # disable hypercolumn
        sp_method="patch",
        ls_init_value=1e-5,
        seg_specific_classifier='Conv'
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))



@register_model
def hr_superformer_stemp4_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_patch7_7_small_224_ls_1e_5(
        fix_drop_path_rate=True,
        pos_embed_position_method="every_stage",
        stem_strides=(4,),
        stem_kernel_sizes=(4,),
        # heads=(6, 6),
        **kwargs,
    )
    
@register_model
def hr_superformer_stemp8_patch7_7_base_512_ls_1e_5(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        depths=(12,),
        stem_strides=(8,),
        stem_channels_list=(768,),
        stem_conv_types=("patch",),
        stem_kernel_sizes=(8,),
        dims=(768,),
        heads=(12, ),
        strides=(1,),
        hypercolumn_indices=(0,),  # disable hypercolumn
        sp_method="sp_cross",
        ls_init_value=1e-5,
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("avgpool",),
        seg_specific_classifier="Conv"


    )
    return SuperformerHR(**_update_params(defaults, **kwargs))

@register_model
def hr_superformer_stemp4_patch7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return (
        hr_superformer_stemp4_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
            depths=(12,),
            dims=(384,),
            heads=(6,),
            strides=(1,),
            sp_sizes=(4,),
            sp_heads=(1,),
            sp_features_init_methods=("avgpool",),
            sp_method="sp_cross",            
            **kwargs,
            
        )
    )
@register_model
def hr_superformer_stemp4_patch7_small_512_ls_1e_5_sp_pe(
    **kwargs,
):
    return (
        hr_superformer_stemp4_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
            depths=(12,),
            dims=(384,),
            heads=(6,),
            strides=(1,),
            sp_sizes=(4,),
            sp_heads=(6,),
            sp_features_init_methods=("avgpool",),
            sp_method="sp_cross",    
            sp_position_embedding_method="depthwise",
            sp_ls_init_value=1e-5,
            merge_multihead_similarities=False,     
            **kwargs,
            
        )
    )
@register_model
def hr_superformer_stemp4_patch7_small_512_2stage(
    **kwargs,
):
    return (
        hr_superformer_stemp4_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
            depths=(12,12),
            dims=(384,384),
            heads=(6,6),
            strides=(1,1),
            sp_sizes=(4,4),
            sp_heads=(6,6),
            sp_features_init_methods=("avgpool",),
            sp_method="sp_cross",    
            sp_position_embedding_method="depthwise",
            sp_ls_init_value=1e-5,
            merge_multihead_similarities=False,     
            **kwargs,
            
        )
    )
@register_model
def hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return (
        hr_superformer_stemp4_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
            **_update_params(
                dict(
                    sp_kwargs={
                        "return_similarities_final": False,
                        "pixel_qkv_method": "linear_shared",
                    },
                    sp_method="sp_cross",
                    sp_features_init_methods=("avgpool", "from_feature"),
                    sp_position_embedding_method="depthwise",
                    sp_ls_init_value=1e-5,
                ),
                **kwargs,
            ),
        )
    )


@register_model
def hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        **_update_params(dict(use_pixel_similarities=True), **kwargs)
    )


@register_model
def hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_heads=[6, 6], merge_multihead_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp4_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        depths=(2, 10), use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_patch7_7_small_224_ls_1e_5(**kwargs):
    return hr_superformer_stemp8_patch7_7_small_224_ls_1e_5(
        **_update_params(dict(heads=(6, 6)), **kwargs)
    )


@register_model
def hr_superformer_stemp8h6_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_patch7_7_small_224_ls_1e_5(
        fix_drop_path_rate=True, pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8h6_patch7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        dims=(384,),
        heads=(6,),
        depths=(12,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(-1,),
        sp_features_init_methods=("avgpool",),
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_patch7_7_base_224_ls_1e_5(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(8,),
        stem_channels_list=(768,),
        stem_conv_types=("patch",),
        stem_kernel_sizes=(8,),
        dims=(768, 768),
        heads=(12, 12),
        strides=(1, 1),
        hypercolumn_indices=(0,),  # disable hypercolumn
        sp_method="patch",
        ls_init_value=1e-5,
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
def hr_superformer_stemp8_patch7_7_base_224_ls_1e_6(**kwargs):
    return hr_superformer_stemp8_patch7_7_base_224_ls_1e_5(ls_init_value=1e-6, **kwargs)


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_patch7_7_base_224_ls_1e_5(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        sp_method="sp_cross",
        sp_features_init_methods=("avgpool", "from_feature"),
        sp_position_embedding_method="depthwise",
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5(
        fix_drop_path_rate=True, pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_head3_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_heads=[3, 3], **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5_fixdroppathrate(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_base_224_ls_1e_5(
        fix_drop_path_rate=True, **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_posembeddw_base_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_patch7_7_base_224_ls_1e_5(
        sp_kwargs={
            "pixel_qkv_method": "identity",
        },
        sp_method="sp_cross",
        sp_features_init_methods=("avgpool", "from_feature"),
        sp_position_embedding_method="depthwise",
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_spsize7_7_spres_nofinal_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_patch7_7_small_224_ls_1e_5(
        **_update_params(
            dict(
                sp_method="sp_cross",
                sp_kwargs={"return_similarities_final": False},
                sp_features_init_methods=("avgpool", "from_feature"),
                sp_position_embedding_method="depthwise",
            ),
            **kwargs,
        )
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_spres_nofinal_posembeddw_small_224_ls_1e_5(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        fix_drop_path_rate=True, **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        class_token_position_method="last_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7depth1_11_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        depths=(1, 11), pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7depth2_10_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        depths=(2, 10), pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7depth6_6_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        depths=(6, 6), pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7depth9_3_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        depths=(9, 3), pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7depth11_1_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        depths=(11, 1), pos_embed_position_method="every_stage", **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate(
        pos_embed_position_method="every_stage",
        class_token_position_method="last_stage",
        **kwargs,
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor_posembedevery(
        heads=(6, 6), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor(
        heads=(6, 6), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        heads=(6, 6), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_280_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    assert kwargs.get("img_size") == 280, kwargs["img_size"]
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_sizes=(5, 5), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_336_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    assert kwargs.get("img_size") == 336, kwargs["img_size"]
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_sizes=(6, 6), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_ls_init_value=1e-5, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        **_update_params(
            dict(use_pixel_similarities=True, sp_ls_init_value=1e-5), **kwargs
        ),
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_head2cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_heads=[2, 2], merge_multihead_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_head3cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_heads=[3, 3], merge_multihead_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_heads=[6, 6], merge_multihead_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_heads=[6, 6], merge_multihead_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        depths=(2, 10), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7depth4_8_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        depths=(4, 8), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_middlepixel_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        use_middle_pixel_features=True, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize14_14_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_448_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    assert kwargs.get("img_size") == 448, kwargs["img_size"]
    return hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize14_14depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_448_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    assert kwargs.get("img_size") == 448, kwargs["img_size"]
    return hr_superformer_stemp8h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        **kwargs
    )


@register_model
def hr_superformer_stemp16h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_448_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    assert kwargs.get("img_size") == 448, kwargs["img_size"]
    return hr_superformer_stemp8h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        stem_kernel_sizes=(16,), stem_strides=(16,), **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_temp_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_temperature_method="individual",
        use_pixel_similarities=True,
        sp_ls_init_value=1e-5,
        **kwargs,
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        heads=(6, 6), **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_spres_nofinal_posembeddw_small_224_ls_1e_5(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        depths=(12,),
        dims=(384,),
        heads=(
            3,
        ),  # NOTE(meijieru): compatible with our previous wrong config, p8h6 for normal.
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("avgpool",),
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_spres_nofinal_posembeddw_small_224_ls_1e_5(
        sp_kwargs={
            "return_similarities_final": True,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeds23_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        sp_position_embedding_method="learnable",
        sp_position_embedding_stride=2,
        sp_sp_position_embedding_stride=3,
        **kwargs,
    )


@register_model
def hr_superformer_stemp8_spsize7_7_head3_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        sp_heads=[3, 3], **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize7_7_head2_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
    **kwargs,
):
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        sp_heads=[2, 2], **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize10_10_pixelqkvidentity_spres_nofinal_posembeddw_small_320_ls_1e_5(
    **kwargs,
):
    assert kwargs.get("img_size") == 320
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        **kwargs
    )


@register_model
def hr_superformer_stemp8_spsize14_14_pixelqkvidentity_spres_nofinal_posembeddw_small_448_ls_1e_5(
    **kwargs,
):
    assert kwargs.get("img_size") == 448, kwargs["img_size"]
    return hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5(
        **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize14_14_pixelqkvidentity_spres_nofinal_posembeddw_small_448_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8_spsize14_14_pixelqkvidentity_spres_nofinal_posembeddw_small_448_ls_1e_5(
        heads=(6, 6),
        fix_drop_path_rate=True,
        pos_embed_position_method="every_stage",
        **kwargs,
    )


@register_model
def hr_superformer_stemp8h6_spsize14_14_pixelqkvidentity_spres_nofinal_posembeddw_small_448_ls_1e_5_fixdroppathrate_clstokencor_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize14_14_pixelqkvidentity_spres_nofinal_posembeddw_small_448_ls_1e_5_fixdroppathrate_posembedevery(
        class_token_position_method="last_stage", **kwargs
    )


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table

    import models

    img_size = 224
    # img_size = 320
    img_size = 448
    model_fns = [
        # hr_superformer_stemp8_patch7_7_small_224_ls_1e_5,
        # hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5,
        # hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery,
        # models.deit_small_patch16_224_ls_1e_5,
        # hr_superformer_stemp4_patch7_7_small_224_ls_1e_5_fixdroppathrate_posembedevery,
        # hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery,
        # models.deit_small_patch32_224_ls1e_5_avgpool,
        # hr_superformer_stemp8h6_patch7_small_224_ls_1e_5_fixdroppathrate_posembedevery,
        # hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery,
        # hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery,
        hr_superformer_stemp16h6_spsize7_7depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_448_ls_1e_5_fixdroppathrate_posembedevery,
    ]

    for model_fn in model_fns:
        model = model_fn(img_size=img_size, drop_path_rate=0.1)
        # https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis
        flops = FlopCountAnalysis(model, torch.rand(1, 3, img_size, img_size))
        print(flop_count_table(flops, max_depth=5))

        total = sum(
            [param.nelement() for param in model.parameters() if param.requires_grad]
        )
        print("Number of parameter: %.4fM" % (total / 1e6))
