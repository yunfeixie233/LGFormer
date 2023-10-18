import hr_superpixel_models
from hr_superpixel_models import _update_params, SuperformerHR
from timm.models.registry import register_model


@register_model
def hr_superformer_tiny_224(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    return SuperformerHR(**kwargs)


@register_model
def hr_superformer_stems8_spsize7_7_small_224(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(2, 2, 2, 1),
        stem_channels_list=(48, 96, 192, 384),
        dims=(384, 384),
        heads=(6, 6),
        strides=(1, 1),
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
def hr_superformer_patch_tiny_224(**kwargs):
    return hr_superformer_tiny_224(sp_method="patch", **kwargs)


@register_model
def hr_superformer_stems8_spsize7_7_tiny_224(**kwargs):
    defaults = dict(
        stem_strides=(2, 2, 2, 1),
        stem_channels_list=(24, 48, 96, 192),
        dims=(192, 192),
        heads=(3, 3),
        strides=(1, 1),
    )
    defaults.update(**kwargs)
    return hr_superformer_tiny_224(
        **defaults,
    )


@register_model
def hr_superformer_stems8_lastconvnext7_spsize7_7_small_224(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(2, 2, 2, 1),
        stem_channels_list=(48, 96, 192, 384),
        stem_kernel_sizes=(3, 3, 3, 7),
        stem_conv_types=("conv", "conv", "conv", "convnext"),
        dims=(384, 384),
        heads=(6, 6),
        strides=(1, 1),
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
# def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(**kwargs):
def hr_superformer_stems8_hyper13_lastsepconv7_spsize7_7_small_224_ls1e_5(
    pretrained=False, **kwargs
):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(2, 2, 2, 1),
        stem_channels_list=(48, 96, 192, 384),
        stem_kernel_sizes=(3, 3, 3, 7),
        stem_conv_types=("conv", "conv", "conv", "sep_conv"),
        dims=(384, 384),
        heads=(6, 6),
        strides=(1, 1),
        hypercolumn_indices=(1, 3),
        ls_init_value=1e-5,
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
def hr_superformer_stems8_hyper13_lastsepconv7_patch7_7_small_224_ls1e_5(**kwargs):
    return hr_superformer_stems8_hyper13_lastsepconv7_spsize7_7_small_224_ls1e_5(
        sp_method="patch", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_lastsepconv3_patch7_7_small_224_ls1e_5(**kwargs):
    return hr_superformer_stems8_hyper13_lastsepconv7_patch7_7_small_224_ls1e_5(
        stem_kernel_sizes=(3, 3, 3, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_base_224(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(2, 2, 1, 2, 1, 1),
        stem_channels_list=(64, 128, 128, 256, 256, 512),
        stem_conv_types=("conv", "conv", "conv", "conv", "conv", "conv"),
        dims=(768, 768),
        heads=(12, 12),
        strides=(1, 1),
        hypercolumn_indices=(2, 5),
        hypercolumn_channels=768,
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
def hr_superformer_stems8_hyper13c768_spsize7_7_base0_224(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(2, 2, 2, 1),
        stem_channels_list=(64, 128, 256, 512),
        stem_conv_types=("conv", "conv", "conv", "conv"),
        dims=(768, 768),
        heads=(12, 12),
        strides=(1, 1),
        hypercolumn_indices=(1, 3),
        hypercolumn_channels=768,
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
def hr_superformer_stems8_hyper13c768_spsize7_7_base0_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13c768_spsize7_7_base0_224(
        sp_features_init_methods=("avgpool", "from_feature"),
        sp_position_embedding_method="depthwise",
        sp_kwargs={
            "return_similarities_final": False,
        },
        ls_init_value=1e-5,
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13c768_spsize7_7_pixelqkvidentity_base0_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13c768_spsize7_7_base0_224(
        sp_features_init_methods=("avgpool", "from_feature"),
        sp_position_embedding_method="depthwise",
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        ls_init_value=1e-5,
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224(**kwargs):
    return hr_superformer_stems8_hyper25c768_spsize7_7_base_224(
        sp_features_init_methods=("avgpool", "from_feature"), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper25c768_patch7_7_base_224_ls1e_5(**kwargs):
    return hr_superformer_stems8_hyper25c768_spsize7_7_base_224(
        sp_method="patch", ls_init_value=1e-5, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_posembeddw(**kwargs):
    return hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224(
        sp_position_embedding_method="depthwise", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_posembeddw(
        **_update_params(
            dict(sp_kwargs={"return_similarities_final": False}),
            **kwargs,
        )
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return (
        hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_nofinal_posembeddw(
            ls_init_value=1e-5, **kwargs
        )
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_pixelqkvidentity_base_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_nofinal_posembeddw_ls1e_5(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_head6_pixelqkvidentity_base_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper25c768_spsize7_7_pixelqkvidentity_base_spres_224_nofinal_posembeddw_ls1e_5(
        sp_heads=[6, 6], **kwargs
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_head3_pixelqkvidentity_base_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper25c768_spsize7_7_pixelqkvidentity_base_spres_224_nofinal_posembeddw_ls1e_5(
        sp_heads=[3, 3], **kwargs
    )


@register_model
def hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_nofinal_posembeddw_test(
    **kwargs,
):
    return hr_superformer_stems8_hyper25c768_spsize7_7_base_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            # "key_expansion": 0.125,
            # "value_expansion": 1.0,
            "key_expansion": 1.0,
            "value_expansion": 1.0,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_patch7_7_small_224(**kwargs):
    return hr_superformer_stems8_spsize7_7_small_224(sp_method="patch", **kwargs)


@register_model
def hr_superformer_stems8_hyper13_patch7_7_small_224(**kwargs):
    return hr_superformer_stems8_patch7_7_small_224(
        hypercolumn_indices=(1, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_patch7_7_small_224_ls1e_5(**kwargs):
    return hr_superformer_stems8_hyper13_patch7_7_small_224(
        ls_init_value=1e-5, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_patch7_small_224_ls1e_5(**kwargs):
    return hr_superformer_stems8_hyper13_patch7_7_small_224(
        depths=(12,),
        dims=(384,),
        heads=(6,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("avgpool",),
        ls_init_value=1e-5,
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_patch10_small_224_ls1e_5(**kwargs):
    assert kwargs.get("img_size") == 320
    return hr_superformer_stems8_hyper13_patch7_small_224_ls1e_5(**kwargs)


@register_model
def hr_superformer_stems8_hyper13_patch10_10_small_320_ls1e_5(**kwargs):
    assert kwargs.get("img_size") == 320
    return hr_superformer_stems8_hyper13_patch7_7_small_224_ls1e_5(**kwargs)


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_small_spres_224(**kwargs):
    return hr_superformer_stems8_spsize7_7_small_224(
        **_update_params(
            dict(
                sp_features_init_methods=("avgpool", "from_feature"),
                hypercolumn_indices=(1, 3),
            ),
            **kwargs,
        )
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_small_spres_224_posembeddw(**kwargs):
    return hr_superformer_stems8_hyper13_spsize7_7_small_spres_224(
        sp_position_embedding_method="depthwise", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_small_spres_224_posembeddw_ls1e_5(**kwargs):
    return hr_superformer_stems8_hyper13_spsize7_7_small_spres_224_posembeddw(
        ls_init_value=1e-5, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_small_spres_224_posembeddw_ls1e_5(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_small_spres_224_posembeddw_ls1e_5(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        hypercolumn_indices=(3,),
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spconvnorm_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_embed_method="conv3x3_norm", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spconv1norm_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_embed_method="conv1x1_norm", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spconvnormact_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_embed_method="conv3x3_norm_act", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spconv_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_embed_method="conv3x3", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_features_init_methods=("avgpool", "avgpool"), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_head2_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_heads=[2, 2], **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_head3_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        sp_heads=[3, 3], **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize14_14_head3_pixelqkvidentity_small_spres_448_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    assert kwargs.pop("img_size") == 448
    return hr_superformer_stems8_hyper13_spsize7_7_head3_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        img_size=448, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize10_10_head3_pixelqkvidentity_small_spres_320_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    assert kwargs.pop("img_size") == 320
    return hr_superformer_stems8_hyper13_spsize7_7_head3_pixelqkvidentity_small_spres_224_nofinal_posembeddw_ls1e_5(
        img_size=320, **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(**kwargs):
    defaults = dict(
        sp_position_embedding_method="learnable",
        sp_position_embedding_stride=2,
    )
    defaults.update(**kwargs)
    return hr_superformer_stems8_spsize7_7_tiny_224(
        **defaults,
    )


@register_model
def hr_superformer_stems8_spsize7_7_tiny_224_spmethod_identity(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224(sp_method="identity", **kwargs)


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_224_spmethod_identity(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_spmethod_identity(
        hypercolumn_indices=(1, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize8_8_tiny_256_posembeds2(**kwargs):
    assert kwargs.pop("img_size") == 256
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(img_size=256, **kwargs)


@register_model
def hr_superformer_stems8_spsize8_8_tiny_spres_256_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize8_8_tiny_256_posembeds2(
        sp_features_init_methods=("avgpool", "from_feature"), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize8_8_tiny_spres_256_posembeds4(**kwargs):
    return hr_superformer_stems8_spsize8_8_tiny_spres_256_posembeds2(
        sp_position_embedding_stride=4, **kwargs
    )


@register_model
def hr_superformer_stems8_patch7_7_tiny_224(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(
        sp_method="patch", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_patch7_7_tiny_224(**kwargs):
    return hr_superformer_stems8_patch7_7_tiny_224(hypercolumn_indices=(1, 3), **kwargs)


@register_model
def hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(
        sp_features_init_methods=("avgpool", "from_feature"), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7depth6_6_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        depths=(6, 6), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7depth11_1_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        depths=(11, 1), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7depth9_3_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        depths=(9, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7depth1_11_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        depths=(1, 11), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        hypercolumn_indices=(1, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(
        hypercolumn_indices=(1, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper123_spsize7_7_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        hypercolumn_indices=(1, 2, 3), **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(**kwargs):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(
        sp_position_embedding_method="depthwise", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_nofinal_posembeddw(**kwargs):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={"return_similarities_final": False},
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_pixelrefinek3_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
        pixel_refine_method="sep_conv_3", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_pixelrefinenoredisualk3_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
        pixel_refine_method="no_residual_sep_conv_3", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_pixelrefinenoredisualk7_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
        pixel_refine_method="no_residual_sep_conv_7", **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeds2(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeds23(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        sp_sp_position_embedding_stride=3,
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeds43(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        sp_sp_position_embedding_stride=3,
        sp_position_embedding_stride=4,
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeds44(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
        sp_sp_position_embedding_stride=4,
        sp_position_embedding_stride=4,
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvlinearshared_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "linear_shared",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvnorm_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            # NOTE(meijieru): check is it conflict with the hypercolumn norm.
            # If we have conv position embed, maybe we can ignore that?
            "pixel_qkv_method": "norm",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvlinearqk0_125_rawv_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            # NOTE(meijieru): check is it conflict with the hypercolumn norm.
            # If we have conv position embed, maybe we can ignore that?
            "pixel_qkv_method": "linear_qk_rawv",
            "key_expansion": 0.125,
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_qk0_125_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "linear",
            "key_expansion": 0.125,
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_qk0_25_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "linear",
            "key_expansion": 0.25,
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_qk0_5_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "linear",
            "key_expansion": 0.5,
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_qk_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "linear",
        },
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_prenormpixel_tiny_spres_224_nofinal_posembeddw(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw(
        pre_norm_pixel=True, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_nofinal_posembeddw_ls1e_5(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_nofinal_posembeddw(
        ls_init_value=1e-5, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_nofinal_posembeddw_ls1e_6(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_nofinal_posembeddw(
        ls_init_value=1e-6, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifier(
    **kwargs,
):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        seg_specific_classifier=True, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifierf3(
    **kwargs,
):
    return (
        hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifier(
            num_blocks_after_seg=3, **kwargs
        )
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifierf2(
    **kwargs,
):
    return (
        hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifier(
            num_blocks_after_seg=2, **kwargs
        )
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifierf1(
    **kwargs,
):
    return (
        hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifier(
            num_blocks_after_seg=1, **kwargs
        )
    )


@register_model
def hr_superformer_stems8_hyper13_spsize14_14_tiny_spres_448_posembeddw(**kwargs):
    assert kwargs.pop("img_size") == 448
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        img_size=448, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize12_12_tiny_spres_384_posembeddw(**kwargs):
    assert kwargs.pop("img_size") == 384
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        img_size=384, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize10_10_tiny_spres_320_posembeddw(**kwargs):
    assert kwargs.pop("img_size") == 320
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        img_size=320, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_7_tiny_iter1_spres_224_posembeddw(**kwargs):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw(
        sp_iter=1, **kwargs
    )


@register_model
def hr_superformer_stems8_hyper13noact_spsize7_7_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeds2(
        hypercolumn_act=False, **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeddw(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_spres_224_posembeds2(
        sp_position_embedding_method="depthwise", **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_tiny_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(
        depths=(12,),
        dims=(192,),
        heads=(3,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("avgpool",),
        **kwargs,
    )


@register_model
def hr_superformer_stems8_hyper13_spsize7_tiny_224_nofinal_posembeddw(**kwargs):
    return hr_superformer_stems8_spsize7_tiny_224_posembeds2(
        hypercolumn_indices=(1, 3),
        sp_position_embedding_method="depthwise",
        sp_kwargs={"return_similarities_final": False},
        **kwargs,
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_tiny_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(
        **_update_params(
            dict(
                depths=(4, 4, 4),
                dims=(192, 192, 192),
                heads=(3, 3, 3),
                strides=(1, 1, 1),
                sp_sizes=(4, 4, 4),
                sp_heads=(1, 1, 1),
                sp_features_init_methods=("avgpool", "avgpool", "avgpool"),
            ),
            **kwargs,
        )
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_7_tiny_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_tiny_224_posembeds2(
        **_update_params(
            dict(
                depths=(3, 3, 3, 3),
                dims=(192, 192, 192, 192),
                heads=(3, 3, 3, 3),
                strides=(1, 1, 1, 1),
                sp_sizes=(4, 4, 4, 4),
                sp_heads=(1, 1, 1, 1),
                sp_features_init_methods=("avgpool", "avgpool", "avgpool", "avgpool"),
            ),
            **kwargs,
        )
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_7_tiny_224_spmethod_identity(**kwargs):
    return hr_superformer_stems8_spsize7_7_7_7_tiny_224_posembeds2(
        sp_method="identity", **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_7_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_7_7_tiny_224_posembeds2(
        sp_features_init_methods=(
            "avgpool",
            "from_feature",
            "from_feature",
            "from_feature",
        ),
        **kwargs,
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_7_tiny_iter1_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_7_7_tiny_spres_224_posembeds2(
        sp_iter=1, **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_7_tiny_iter1_spres_224_posembeddw(**kwargs):
    return hr_superformer_stems8_spsize7_7_7_7_tiny_iter1_spres_224_posembeds2(
        sp_position_embedding_method="depthwise", **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_7_tiny_224_posembeds2(
        sp_features_init_methods=("avgpool", "from_feature", "from_feature"), **kwargs
    )


@register_model
def hr_superformer_stems8_spsize7_7_7_tiny_iter1_spres_224_posembeds2(**kwargs):
    return hr_superformer_stems8_spsize7_7_7_tiny_spres_224_posembeds2(
        sp_iter=1, **kwargs
    )


@register_model
def hr_superformer_stems8_patch7_tiny_224(**kwargs):
    return hr_superformer_stems8_spsize7_tiny_224_posembeds2(
        sp_method="patch", **kwargs
    )


@register_model
def hr_superformer_tiny_224_posembeds2(**kwargs):
    return hr_superformer_tiny_224(
        sp_position_embedding_method="learnable",
        sp_position_embedding_stride=2,
        **kwargs,
    )


@register_model
def hr_superformer_tiny_spres_224_posembeds2(**kwargs):
    return hr_superformer_tiny_224_posembeds2(
        sp_features_init_methods=("avgpool", "from_feature"), **kwargs
    )


@register_model
def hr_superformer_tiny_iter1_spres_224_posembeds2(**kwargs):
    return hr_superformer_tiny_spres_224_posembeds2(sp_iter=1, **kwargs)


@register_model
def hr_superformer_tiny_iter1_224_posembeds2(**kwargs):
    return hr_superformer_tiny_224_posembeds2(sp_iter=1, **kwargs)


@register_model
def hr_superformer_tiny_spsize7_7_224_posembeds2(**kwargs):
    return hr_superformer_tiny_224_posembeds2(sp_sizes=(8, 4), **kwargs)


@register_model
def hr_superformer_tiny_224_nofinal_posembeds2(**kwargs):
    return hr_superformer_tiny_224_posembeds2(
        sp_kwargs={"return_similarities_final": False},
        **kwargs,
    )


@register_model
def hr_superformer_stems8l3_patch7_7_tiny_224(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()
    defaults = dict(
        stem_strides=(2, 2, 2),
        # NOTE(meijieru): too large for tiny
        stem_channels_list=(48, 96, 192),
        stem_conv_types=("conv", "conv", "conv"),
        dims=(192, 192),
        heads=(3, 3),
        strides=(1, 1),
        hypercolumn_indices=(2,),  # disable hypercolumn
        sp_method="patch",
    )
    return SuperformerHR(**_update_params(defaults, **kwargs))


@register_model
def hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superpixel_models.hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superpixel_models.hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_fixpixel_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    module = hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_pixel_features_update_method="fixed", **kwargs
    )
    return module


@register_model
def hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_fixpixel_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_pixel_features_update_method="fixed", **kwargs
    )


@register_model
def hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_fixpixel_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp4_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_pixel_features_update_method="fixed", use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery(
    **kwargs,
):
    return hr_superformer_stemp4_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery(
        sp_ls_init_value=None, **kwargs
    )
