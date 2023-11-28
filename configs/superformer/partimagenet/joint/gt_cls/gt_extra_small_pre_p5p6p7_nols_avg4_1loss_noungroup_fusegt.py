_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    use_final_group_cls = True,
    group_pos = ((),(3,4,5,),()),
    ungroup_enable = (False,False,False,),
    use_gt_fuse = True,                        
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',draw = True, interval =10))