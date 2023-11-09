_base_ = [
    '../_base_/datasets/partimagenet_158.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
# model_cfg
<<<<<<< HEAD
num_classes = 12
# num_classes = 11

<<<<<<< Updated upstream
=======
num_classes = 159
>>>>>>> 74f5813f1fa7e98c18045caa597b53036c2aeacc
=======
>>>>>>> Stashed changes
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    decode_head=dict(
        type='MaskFormerHead',
        in_channels=[256, 512, 1024,
                     2048],  # input channels of pixel_decoder modules
        feat_channels=256,
        in_index=[0, 1, 2, 3],
        num_classes=num_classes,
        out_channels=256,
        num_queries=100,
        pixel_decoder=dict(
            type='mmdet.PixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # DetrTransformerDecoder
            return_intermediate=True,
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.1,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=1.0),
                    dict(
                        type='mmdet.FocalLossCost',
                        weight=20.0,
                        binary_input=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=1.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05),
    accumulative_counts= 1,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }
        ))

total_iter = 100000
param_scheduler = [
    dict(
            type='MultiStepLR',
            begin=0,                     # 从第0个epoch开始
            end=total_iter,            # 在总训练周期结束时停止更新学习率
            by_epoch=False,               # 通过epoch来更新学习率
            milestones=[int(total_iter * 0.9), int(total_iter * 0.95)],  # 在第90个和第95个epoch降低学习率
            gamma=0.1,                   # 学习率衰减因子
            verbose=False                # 设置为True以打印每次更新的学习率
    )
]

# In MaskFormer implementation we use batch size 2 per GPU as default
<<<<<<< Updated upstream
<<<<<<< HEAD
train_dataloader = dict(batch_size=16, num_workers=8)
=======
train_dataloader = dict(batch_size=8, num_workers=4)
>>>>>>> 74f5813f1fa7e98c18045caa597b53036c2aeacc
=======
train_dataloader = dict(batch_size=16, num_workers=8)
>>>>>>> Stashed changes
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iter, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

