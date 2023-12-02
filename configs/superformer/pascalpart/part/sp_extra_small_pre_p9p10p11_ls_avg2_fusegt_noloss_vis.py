_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (2,2,2,),
    group_init_kernel_sizes = (2,2,2,),
    group_layers = {0:256,1:256,2:256},         
))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook'))
    visualization=dict(type='SegVisualizationHook',draw = True, interval = 1))