_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    output_dir = "/root/autodl-tmp/vis",
    vis_spgt = True,
    vis_sp_id = True, 
    vis_gt = True,      

))
accumulative_counts = 2
total_iter=25000 * accumulative_counts
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
            'stem':dict(lr_mult=0.1),
            'sp_init':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.0':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1._sp_qkv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1._pixel_qkv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.sp_pos_conv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.pixel_pos_conv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.sp_ls1.':dict(lr_mult=0.1),
            'stages.0.sp_project':dict(lr_mult=0.1),
            'stages.0.blocks':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.0':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1._sp_qkv.':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1._pixel_qkv.':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1.sp_pos_conv.':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1.pixel_pos_conv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.sp_ls1.':dict(lr_mult=0.1),         
            'stages.1.blocks':dict(lr_mult=0.1)},    
        ))

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