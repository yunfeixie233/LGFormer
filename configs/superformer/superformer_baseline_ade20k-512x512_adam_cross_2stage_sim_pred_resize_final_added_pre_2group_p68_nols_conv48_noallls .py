_base_ = [
    '../_base_/default_runtime.py', 
    '../_base_/datasets/ade20k.py',
    './superformer_baseline.py'
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
    resize_version = 'v2',
    use_group_token = 'mix',

    arch_settings = {
        'embed_dims': 384,
        'patch_size': 8,
        'window_size': 2,
        'num_layers': 2,
        'num_heads': 2,
        'num_group_heads': 2,
        'num_group_forward_heads': 2,
        'num_ungroup_heads': 2,
        'ffn_ratio': 4.,
        'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),
        'mlpmixer_depth': 1,
        'group_layers': {0:64,1:16,},
        'drop_path_rate': 0.2,
        'group_projector_methonds':'linear',
        'association_embedding':False,
        'group_token_init_method':'conv',
        "init_strides":(4,(2,4,),),
        'init_kernel_sizes':(4,(2,4,),),
        'merge_pos':((),(6,8,),()),
        'all_ls' : False,
        'ls_init_value':0.,            
    },
    
),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)))

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
            'ls':dict(decay_mult=0.),
            'ln':dict(decay_mult=0.),
            'bn':dict(decay_mult=0.),
            'stem.conv_layers.0.1':dict(decay_mult=0.),
            'sp_project.1':dict(decay_mult=0.),
            'sp_init.1':dict(decay_mult=0.),
            'seg_norm': dict(decay_mult=0.),
            'gamma2':dict(decay_mult=0.),
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
