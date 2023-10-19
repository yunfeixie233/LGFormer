_base_ = [
    '../../../projects/ViTDet/configs/lsj-100e_coco-instance.py',
    '../../_base_/models/mask-rcnn_r50_fpn.py',    
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
custom_imports = dict(imports=['projects.ViTDet.vitdet'])

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,        
        type='SuperformerBottleNeck_ori',
        classification_feature = 'backbone_extralayer',
        sp_global_init_method="lift_avgpool",
        sp_method="sp_cross_asymmetry",
        img_size=(1024,1024),
        seg_specific_classifier="Linear",
        seg_num_classes= 150,
        depths=(2,10,0,),
        dims=(384,384,384,),
        heads=(6,6,-1,),
        strides=(1,1,1,),
        sp_sizes=(4,4,4,),
        sp_heads=(2,2,1,),
        stem_kernel_sizes=(4,),
        stem_conv_types=("conv",),
        stem_channels_list=(64,),
        stem_strides=(4,),
        use_stem=True,
        sp_kwargs={
            "return_similarities_final": False,
        },
        pos_embed_position_method="every_stage",
        sp_features_init_methods=("from_feature","from_feature","from_feature",),
        sp_position_embedding_method="depthwise",
        sp_iter=2,
        drop_path_rate = 0.1,
        ls_init_value = 1e-5,),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=384,
        in_channels=[96, 192, 384, 384],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    _delete_ = True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm1': dict(decay_mult=0.),
            'norm2': dict(decay_mult=0.),
            'ls':dict(decay_mult=0.),
            'ln':dict(decay_mult=0.),
            'bn':dict(decay_mult=0.),
            'stem.conv_layers.0.1':dict(decay_mult=0.),
            'sp_project.1':dict(decay_mult=0.),
            'sp_init.1':dict(decay_mult=0.),
            'seg_norm': dict(decay_mult=0.),
            'gamma': dict(decay_mult=0.),
            'reweight': dict(decay_mult=0.),
        }))



log_processor = dict(by_epoch=False)
find_unused_parameters=True
custom_hooks = [dict(type='Fp16CompresssionHook')]
