_base_ = [
    '../../../../_base_/default_runtime.py', 
    '../../../../_base_/datasets/partimagenet_object.py',
    '../../../superformer_baseline.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
    img_size=(512,512),
    classification_feature = 'superpixel_extralayer',
    resize_similarity =  True,          
    depths=(2,10,0,),
    dims=(384,384,384,),
    heads=(6,6,-1,),
    strides=(1,1,1,),
    sp_sizes=(4,4,4,),
    sp_heads=(2,2,1,),
    sp_features_init_methods=("from_feature","from_feature","from_feature",),
    ls_init_value = 1e-5,
    seg_num_classes = 12,
),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05),
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
            'stages.0':dict(lr_mult=0.1),
            'stages.1':dict(lr_mult=0.1),
        }))

total_iter = 50000
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
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iter, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True,interval=1))


find_unused_parameters=True
