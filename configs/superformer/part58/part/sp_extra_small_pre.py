_base_ = [
    '../../../_base_/default_runtime.py', 
    '../../../_base_/datasets/pascal_voc12_part.py',
    '../../superformer_baseline.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
seg_num_classes = 58
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
    seg_num_classes=seg_num_classes,    
    sp_features_init_methods=("from_feature","from_feature","from_feature",),
    ls_init_value = 1e-5,
),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)))

accumulative_counts = 2
total_iter=40000 * accumulative_counts
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='SGD',
    lr=0.007,
    momentum=0.9,
    weight_decay=1e-4,),
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
        type='PolyLR',
        power=0.9,
        begin=0,
        end=total_iter,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iter, val_interval=1)
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
