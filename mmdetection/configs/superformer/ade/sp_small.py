
_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/ade20k_instance.py',
    '../../_base_/schedules/schedule_1x.py'
     
]
crop_size = (512, 512)
num_classes = 100
model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=512),
    backbone=dict(
        _delete_=True,        
        type='SuperformerBottleNeck_ori',
        classification_feature = 'backbone_extralayer',
        sp_global_init_method="lift_avgpool",
        sp_method="sp_cross_asymmetry",
        img_size=(512,512),
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
    neck=dict(in_channels=[64,64,384,384,]),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),    
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 4, 16, 16]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 4, 16, 16]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))


train_cfg = dict(
    _delete_ = True,
    type='IterBasedTrainLoop', max_iters=160000, val_interval=100)


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optimizer_config=dict(grad_clip=dict(max_norm=10, norm_type=2))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    _delete_ = True,
    type='AdamW',
    lr=0.000006,
    betas=(0.9, 0.999),
    weight_decay=0.01),

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


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=1000))
log_processor = dict(by_epoch=False)
find_unused_parameters=True
