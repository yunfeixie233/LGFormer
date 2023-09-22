norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='pseudo',),
    decode_head=dict(
        type='SuperformerBottleNeck_ori',
        sp_global_init_method="lift_avgpool",
        sp_method="sp_cross_asymmetry",
        img_size=(640,640),
        seg_specific_classifier="Linear",
        seg_num_classes= 150,
        depths=(2,10,0,),
        dims=(384,384,384,),
        heads=(2,2,1,),
        strides=(1,1,1,),
        sp_sizes=(4,4,4,),
        sp_heads=(2,2,2,),
        stem_kernel_sizes=(3,3,3,),
        stem_conv_types=("conv","conv","conv",),
        stem_channels_list=(16,32,64,),
        stem_strides=(2,2,1,),
        use_stem=True,
        sp_kwargs={
            "return_similarities_final": False,
        },
        pos_embed_position_method="every_stage",
        sp_features_init_methods=("from_feature","from_feature","from_feature",),
        sp_position_embedding_method="depthwise",
        sp_iter=2,
        loss_decode=[
            dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(640, 640)))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm1': dict(decay_mult=0.),
            'norm2': dict(decay_mult=0.),
            # 'backbone':dict(lr_mult=0.1)
        }))


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters=True

