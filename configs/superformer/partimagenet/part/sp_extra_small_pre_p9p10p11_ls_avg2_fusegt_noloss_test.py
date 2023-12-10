_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg2_fusegt_noloss.py',
]

model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    output_dir = "/data1/yunfei/sp_extra_small_pre_p9p10p11_ls_avg2_fusegt_noloss_test",
    vis_gt = True,     
),
 test_cfg=dict(mode='whole'))
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',draw = False)
    )
    # visualization=dict(type='SegVisualizationHook'))
test_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]   
# test_dataloader =val_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline))    

test_dataloader =val_dataloader = dict(
    dataset=dict(

        pipeline=test_pipeline))