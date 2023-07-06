_base_ = [
    '../_base_/models/stvit_neck.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='SpformerNeck',
       embed_dim=[256, 256, 256, 256],
                    depths=[4, 2, 1, 1],
                    num_heads=[2,2,2, 2],
                    n_iter=[4, 4, 4, 4], 
                    stoken_size=[4, 4, 4, 4],
                    projection=256,                    
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0,
                    drop_path_rate=0.3, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, True, True],
                    init_values=1e-5,
                    seg=False,
                    downsample=False,
                    in_chans=256),
    decode_head=dict(
        type='SpformerHead',
        in_channels=256,
       channels=512,
       num_classes=150,
       stoken_size=[4, 4, 4, 4])
)
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01,momentum=0.9,weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            # 'absolute_pos_embed': dict(decay_mult=0.),
            # 'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=5000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5000,
        end=160000,
        by_epoch=False,
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
find_unused_parameters=True

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
