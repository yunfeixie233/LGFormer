_base_ = [
    './gt_extra_branchp3_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    depths_obj = (None,1,-1,),
    group_pos = [[],[9,],[]], 
    group_init_strides = (4,),
    group_init_kernel_sizes = (4,),
    group_layers = (64,),   
    group_token_init_method = ('avgpool',),        
    onlyobj_merge_layer = True,
    part_cls_method = "cls_first",
    stem_kernel_sizes=(3, 3),
    stem_conv_types=("conv", "conv"),
    stem_channels_list=(32, 64),
    stem_strides=(2, 2),   
    shared_blocks = False,     
    shared_merge_layer = False, 
    detach_obj_feature = True,    
    resize_similarity_obj = False,
    resize_similarity_part = False,
    )
)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(draw= False, type='SegVisualizationHook'))