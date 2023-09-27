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
        img_size=(512,512),
        seg_specific_classifier="Linear",
        seg_num_classes= 150,
        depths=(12,),
        dims=(384,),
        heads=(6,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(2,),
        stem_kernel_sizes=(4,),
        stem_conv_types=("conv",),
        stem_channels_list=(64,),
        stem_strides=(4,),
        use_stem=True,
        sp_kwargs={
            "return_similarities_final": False,
        },
        pos_embed_position_method="every_stage",
        sp_features_init_methods=("from_feature",),
        sp_position_embedding_method="depthwise",
        sp_iter=2,
        drop_path_rate = 0.1,        
        loss_decode=[
            dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(640, 640)))




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

