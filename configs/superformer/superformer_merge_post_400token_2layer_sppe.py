_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
    '../_base_/datasets/ade20k_640x640.py',
]
custom_imports = dict(imports='mmpretrain', allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    backbone=dict(
        type='pseudo',),
    decode_head=dict(
        type='SuperformerBottleNeck',
        # sampler=dict(type='OHEMPixelSampler_v2',ratio=0.5),
        sp_global_init_method="lift_avgpool",
        sp_method="sp_cross_asymmetry",
        img_size=(640,640),
        seg_specific_classifier="Linear",
        seg_num_classes= 150,
        depths=(2,10,0,),
        dims=(192,192,192,),
        heads=(2,2,1,),
        strides=(1,1,1,),
        sp_sizes=(4,4,4,),
        sp_heads=(2,2,1,),
        cross_only=(False,False,True,),
        stem_kernel_sizes=(3, 3, 3),
        stem_conv_types=("conv", "conv", "conv"),
        stem_channels_list=(16, 32, 64),
        stem_strides=(2, 2, 1),
        use_stem=True,
        sp_kwargs={
            "return_similarities_final": False,
        },
        pos_embed_position_method="every_stage",
        sp_features_init_methods=("from_feature","from_feature","from_feature",),
        sp_position_embedding_method="learnable",
        sp_position_embedding_stride=4,
        sp_iter=2,
        num_group_tokens = (400,400,),
        num_output_groups = (400,400,),
        group_token_init_methods=("learnable","learnable",),
        group_token_embedding_method=(None,None,),
        token_strides =(4,4,),
        token_channels_list=(384,384,),
        depths_merge = (6,6,),
        merge_heads = (6,6,),
        hard_assignment = (False,False,),
        # vis_token = 'all_groups',
        # output_dir = '/root/autodl-tmp/superformer_merge_post_400token_2layer_2dpe',
        vis_featuremap = False,
        vis_sp_logits = False,
        group_projector_from_sp_feature=True,
        # downsample_first = True,
        loss_decode=[
            dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # dict(
            #     type='OhemCrossEntropy',
            #     thres=0.5,
            #     min_kept=131072,
            #     loss_weight=1.0),
        # dict(type='FocalLoss', 
        #      loss_name='loss_focal',
        #         use_sigmoid=True,
        #         gamma=2.0,
        #         alpha=0.5,
        #         reduction='mean',
        #         loss_weight=3.0),
    ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

bgr_mean = data_preprocessor['mean'][::-1]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 640) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='mmpretrain.datasets.transforms.AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm1': dict(decay_mult=0.),
            'norm2': dict(decay_mult=0.),
            # 'backbone':dict(lr_mult=0.1)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=2000,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]



train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=2000)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


find_unused_parameters=True

