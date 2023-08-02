from typing import MutableMapping, Optional, Tuple, Union
import einops
import warnings

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from superpixel.superpixel_cross_attention_ops import LayerNorm2DChannelOnly

import superpixel.superpixel_transformer as st
from superpixel.superpixel_transformer import SuperPixelStem, SimilaritiesHead
from superpixel import superpixel_ops


def make_superformer_cfg():
    cfg = _cfg()
    cfg['first_conv'] = 'patch_embed.conv_stem.conv_layers.0.0.weight'
    return cfg


class Superformer(VisionTransformer):

    def __init__(self, superpixel_kwargs={}, token_specific_classifier=False,
                 seg_specific_classifier=False, seg_num_classes=-1, seg_block_idx=-1,
                 similarities_classifier=False, similarities_kernel_size=3, **kwargs):
        super().__init__(
            **kwargs,
            embed_layer=partial(
                SuperPixelStem, **
                superpixel_kwargs)  # pyright: ignore [reportGeneralTypeIssues]
        )

        assert sum([token_specific_classifier, seg_specific_classifier]) <= 1

        norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))

        self.token_specific_classifier = token_specific_classifier
        if self.token_specific_classifier:
            # NOTE(meijieru): the token features doesn't use avgpool, so we skip
            # the fc_norm.
            self.token_head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.seg_specific_classifier = seg_specific_classifier
        self.seg_num_classes = seg_num_classes
        # [0, self.seg_block_idx) are the blocks for segmentation
        assert seg_block_idx < 0
        self.seg_block_idx = len(self.blocks) + 1 + seg_block_idx
        if self.seg_specific_classifier:
            assert seg_num_classes > 0
            # NOTE(meijieru): the token features doesn't use avgpool, so we skip
            # the fc_norm.
            self.seg_head = nn.Linear(self.embed_dim, self.seg_num_classes) if self.seg_num_classes > 0 else nn.Identity()
            if self.seg_block_idx != len(self.blocks):
                self.seg_norm = norm_layer(self.embed_dim)
            print(f'use block{self.seg_block_idx} for seg')

        self.similarities_classifier = similarities_classifier
        if self.similarities_classifier:
            assert self.patch_embed.tokenization_method != 'patch'
            self.similarities_head = SimilaritiesHead(
                    self.patch_embed.pixel_output_shape,
                    self.patch_embed.superpixel_shape,
                    self.embed_dim, self.num_classes,
                    kernel_size=similarities_kernel_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def tokenization_info(self) -> Optional[MutableMapping[str, torch.Tensor]]:
        return self.patch_embed.tokenization_info  # pyright: ignore [reportGeneralTypeIssues]

    def visualize_superpixel(self, resize_similarities: bool = True):
        return st.visualize_superpixel(self.tokenization_info, self.patch_embed, resize_similarities=resize_similarities)

    def no_weight_decay(self):
        no_weight_decay = super().no_weight_decay()
        no_weight_decay = no_weight_decay.union({
            'patch_embed.superpixel_tokenize.init_superpixel_queries'
        })
        for name, _ in self.named_parameters():
            if 'pos_embed' in name:
                no_weight_decay.add(name)
        print(f'no_weight_decay: {no_weight_decay}')
        return no_weight_decay

    def forward_blocks_range(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError()
        else:
            for i in range(start, end):
                x = self.blocks[i](x)
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        # [0, seg_block_idx) are the blocks for segmentation
        x_seg = self.forward_blocks_range(x, 0, self.seg_block_idx)
        x_cls = self.forward_blocks_range(x_seg, self.seg_block_idx, len(self.blocks))
        return x_cls, x_seg

    def forward(self, x: torch.Tensor, generate_seg: bool = False,
                generate_simil_pred: bool = False,
                return_pixel_logits: bool = True,
                seg_stride: int = 2) -> Union[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        x_cls, x_seg = self.forward_features(x)
        logits = self.forward_head(self.norm(x_cls))
        ret = {'cls': logits}
        if generate_seg:
            sp_logits, pixel_logits = self.forward_segmentation(
                    x_seg, return_pixel_logits, stride=seg_stride)
            ret.update({ 'seg': pixel_logits, 'sp_cls': sp_logits})
        if generate_simil_pred:
            similarities = self.patch_embed.similarities_final
            similarities = st.prepare_similarities(self.patch_embed, similarities, resize=False)
            ret['cls_similarities'] = self.similarities_head(similarities)
        return ret

    def forward_segmentation(self, x: torch.Tensor,
                             return_pixel_logits: bool = True,
                             stride: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.seg_block_idx == len(self.blocks):
            # For compatiable with old models
            # NOTE(meijieru): self.norm is shared between cls & seg
            x = self.norm(x)
            x = x[:, self.num_prefix_tokens:]
        else:
            # NOTE(meijieru): norm only for seg part
            x = self.seg_norm(x[:, self.num_prefix_tokens:])
        b, num, c = x.shape
        x = x.reshape(b * num, c)

        if self.token_specific_classifier:
            sp_logits = self.token_head(x)
        elif self.seg_specific_classifier:
            sp_logits = self.seg_head(x)
        else:
            print('use image classifier for superpixels')
            x = self.fc_norm(x)
            sp_logits = self.head(x)

        sh, sw = self.patch_embed.superpixel_shape
        sp_logits = sp_logits.view(b, sh, sw, -1).permute(0, 3, 1, 2)

        if isinstance(self.patch_embed, SuperPixelStem) and return_pixel_logits:
            similarities = self.patch_embed.similarities_final
            if similarities is None:
                raise ValueError()
            similarities = st.prepare_similarities(self.patch_embed, similarities, resize=True, stride=stride)
            similarities = similarities.softmax(1)
            similarities = einops.rearrange(similarities, 'b n sh ph sw pw -> b n (sh ph) (sw pw)')
            pixel_logits = superpixel_ops.expand_superpixel_features(
                sp_logits, similarities)
        else:
            pixel_logits = None

        return sp_logits, pixel_logits


@register_model
def superformer_small_patch16_224(pretrained=False, **kwargs):
    model = Superformer(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_patch16_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        superpixel_iter=1,
    )
    model = Superformer(patch_size=16,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_patch16_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        superpixel_iter=2,
    )
    model = Superformer(patch_size=16,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter1_patch16_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        superpixel_iter=1,
    )
    model = Superformer(patch_size=16,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_patch32_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(2, 3),
        superpixel_iter=2,
        pixel_stride=8,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter1_patch32_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(2, 3),
        superpixel_iter=1,
        pixel_stride=8,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224_baseline(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
        tokenization_method='patch',
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_querylearnable(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        superpixel_features_init_method='learnable',
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224_querylearnable(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
        superpixel_features_init_method='learnable',
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_querylearnable_notoken(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        superpixel_features_init_method='learnable',
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_notoken(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_notoken_dualnorm(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        proj_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_notoken_prepostposnorm(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        pre_norm=True,
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_notoken_projnorm(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        proj_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_notoken_prenorm(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224_notoken_dualnorm(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        proj_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224_notoken_prenorm_dualpath(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224_prenorm_dualpath(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_hypercolumn3_patch32_224_notoken_prenorm_dualpath(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=1,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_hypercolumn3_patch32_224_notoken_prenorm_dualpath_querylearnable(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 2, 3),
        superpixel_iter=2,
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
        superpixel_features_init_method='learnable',
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter2_stride8_patch32_224_notoken_prenorm_dualpath(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=None,
        hypercolumn_indices=(3,),
        superpixel_iter=2,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notoken_prenorm_dualpath(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        superpixel_iter=2,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenorm_dualpath_posembeds2_tokenclassifier(pretrained=False, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenorm_dualpath_posembeds2(token_specific_classifier=True, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notoken_prenorm_dualpath_posembeds2(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        superpixel_iter=2,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
        position_embedding_method="learnable",
        position_embedding_stride=2,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenorm_dualpath_posembeds2(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        superpixel_iter=2,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
        position_embedding_method="learnable",
        position_embedding_stride=2,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=False,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model


@register_model
def superformer_tiny_iter1_stride8_hypercolumn13_patch32_224_prenorm(
        pretrained=False, superpixel_kwargs_update=None, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        superpixel_iter=1,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    if superpixel_kwargs_update is not None:
        superpixel_kwargs.update(superpixel_kwargs_update)
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplementedError()
    return model


@register_model
def superformer_tiny_iter1_stride8_hypercolumn13_patch32_224_prenorm_similclassifierk1(
        **kwargs):
    return superformer_tiny_iter1_stride8_hypercolumn13_patch32_224_prenorm(
        similarities_classifier=True, similarities_kernel_size=1, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_similclassifierk1(
        **kwargs):
    return superformer_tiny_iter1_stride8_hypercolumn13_patch32_224_prenorm(
        similarities_classifier=True,
        similarities_kernel_size=1,
        superpixel_kwargs_update={"superpixel_iter": 2},
        **kwargs)


# base
@register_model
def superformer_tiny_iter1_stride8_hypercolumn13_patch32_224(superpixel_kwargs_update=None, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    if kwargs.pop('pretrained', False):
        raise NotImplementedError()
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        strides=[2, 2, 2, 1],
        pixel_stride=8,
    )
    if superpixel_kwargs_update is not None:
        superpixel_kwargs.update(superpixel_kwargs_update)
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    return model


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(superpixel_kwargs_update=None, **kwargs):
    superpixel_kwargs_update_final = dict(
        superpixel_iter=2,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="sp_cross",
        position_embedding_method="learnable",
        position_embedding_stride=2,
    )
    superpixel_kwargs_update_final.update(superpixel_kwargs_update or {})
    return superformer_tiny_iter1_stride8_hypercolumn13_patch32_224(superpixel_kwargs_update_final, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(
            superpixel_kwargs_update={
                "pre_norm_layer": partial(LayerNorm2DChannelOnly, eps=1e-6),
                **(superpixel_kwargs_update or {})
            }, **kwargs)

@register_model
def superformer_tiny_iter1_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
            superpixel_kwargs_update=dict(
                superpixel_iter=1,
                **(superpixel_kwargs_update or {})
                ), **kwargs
            )

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2_segclassifier(**kwargs):
    warnings.warn('Wrong pre_norm, only for compatibility')
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(
            seg_specific_classifier=True, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
            seg_specific_classifier=True, **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenormchtrue_spcross_posembeds2_segclassifier(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier(
            class_token=False, global_pool="avg", **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier_l11(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier(seg_block_idx=-2, **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcrossnofinal_posembeds2_segclassifier_l11(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier_l11(superpixel_kwargs_update={
                "return_similarities_final": False,
                **(superpixel_kwargs_update or {})
            }, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier_l10(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier(seg_block_idx=-3, **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenormchtrue_spcross_posembeds2_segclassifier_l10(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier_l10(class_token=False, global_pool="avg", **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcrossnofinal_posembeds2_segclassifier_l10(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier_l10(superpixel_kwargs_update={
                "return_similarities_final": False,
                **(superpixel_kwargs_update or {})
            }, **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier_l9(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcross_posembeds2_segclassifier(seg_block_idx=-4, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormchtrue_spcrossnofinal_posembeds2_segclassifier(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(superpixel_kwargs_update={
                "return_similarities_final": False,
                **(superpixel_kwargs_update or {})
            }, seg_specific_classifier=True, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenormch_spcross_posembeds2_segclassifier(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
            seg_specific_classifier=True, class_token=False, global_pool="avg" , **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcrossnofinal_posembeds2_tokenclassifier(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(
            superpixel_kwargs_update={
                "pre_norm_layer": partial(LayerNorm2DChannelOnly, eps=1e-6),
                "return_similarities_final": False,
                **(superpixel_kwargs_update or {})
            }, token_specific_classifier=True, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
            superpixel_kwargs_update={
                "position_embedding_method": "none",
                **(superpixel_kwargs_update or {})
            }, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_identity(
        superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
        superpixel_kwargs_update={
            "position_embedding_method": "none",
            "tokenization_method": "identity",
            **(superpixel_kwargs_update or {})
        },
        **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_dualpath_posembeds2(
        superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
        superpixel_kwargs_update={
            "tokenization_method": "dual_path",
            **(superpixel_kwargs_update or {})
        },
        **kwargs)


@register_model
def superformer_tiny_iter3_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
        superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
        superpixel_kwargs_update={
            "superpixel_iter": 3,
            **(superpixel_kwargs_update or {})
        },
        **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcrossfixpixel_posembeds2(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_spcross_posembeds2(
            superpixel_kwargs_update={
                "pixel_features_update_method": "fixed",
                **(superpixel_kwargs_update or {})
            }, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenormch_baseline(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(
            superpixel_kwargs_update={
                "pre_norm_layer": partial(LayerNorm2DChannelOnly, eps=1e-6),
                "tokenization_method": 'patch',
                **(superpixel_kwargs_update or {})
            }, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_dualnormch_spcross_posembeds2(superpixel_kwargs_update=None, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(
            superpixel_kwargs_update={
                "pre_norm_layer": partial(LayerNorm2DChannelOnly, eps=1e-6),
                "proj_norm_layer": partial(LayerNorm2DChannelOnly, eps=1e-6),
                **(superpixel_kwargs_update or {})
            }, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notokencor_prenorm_spcross_posembeds2(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(class_token=False, global_pool="avg" , **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2_similclassifierk1(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(similarities_classifier=True, similarities_kernel_size=1, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcrossnofinal_posembeds2(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2(
            superpixel_kwargs_update={"return_similarities_final": False}, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcrossnofinal_posembeds2_tokenclassifier(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcrossnofinal_posembeds2(token_specific_classifier=True, **kwargs)

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcrossnofinal_posembeds2_similclassifierk1(**kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_spcross_posembeds2_similclassifierk1(
            superpixel_kwargs_update={"return_similarities_final": False}, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        superpixel_iter=2,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method="dual_path",
        position_embedding_method="learnable",
        position_embedding_stride=2,
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model

@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2_tokenclassifier(pretrained=False, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2(token_specific_classifier=True, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2_similclassifier(pretrained=False, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2(similarities_classifier=True, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2_similclassifierk1(pretrained=False, **kwargs):
    return superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_prenorm_dualpath_posembeds2(similarities_classifier=True, similarities_kernel_size=1, **kwargs)


@register_model
def superformer_tiny_iter2_stride8_hypercolumn13_patch32_224_notoken_prenorm_baseline(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    superpixel_kwargs = dict(
        stem_channels_list=(24, 48, 96, 192),
        hypercolumn_dim=192,
        hypercolumn_indices=(1, 3,),
        superpixel_iter=2,
        strides=[2, 2, 2, 1],
        pixel_stride=8,
        pre_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tokenization_method='patch',
    )
    model = Superformer(patch_size=32,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        superpixel_kwargs=superpixel_kwargs,
                        class_token=True,
                        global_pool="avg",
                        **kwargs)
    model.default_cfg = make_superformer_cfg()
    if pretrained:
        raise NotImplemented()
    return model
