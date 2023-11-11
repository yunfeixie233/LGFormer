_base_ = [
    './gt_extra_small_pre_p9_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (4,4,4,),
    group_init_kernel_sizes = (4,4,4,),
    group_layers = {0:64,1:64,2:64},   
    group_token_init_method = ('avgpool','avgpool','avgpool',), 
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=0.7,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_1', use_sigmoid=False, loss_weight=0.1,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_2', use_sigmoid=False, loss_weight=0.1,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_3', use_sigmoid=False, loss_weight=0.1,reduction='mean',),            
            ],         
))

accumulative_counts = 4
total_iter=10000 * accumulative_counts
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05),
    accumulative_counts=accumulative_counts,        
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
    type='IterBasedTrainLoop', max_iters=total_iter, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters=True
