_base_ = [
    '../_base_/models/regproxy/regproxy-s16.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'    
    # '../_base_/schedules/adamw+poly-power_1+lr_6e-5+wd_0.01+iter_160k.py'
]
global_channels = 384

model = dict(
    backbone=dict(
        out_indices=[11,]),
    decode_head=dict(
        in_channels=(global_channels,),
        in_index=(-1,),
        num_classes=150))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            '.ln': dict(decay_mult=0.),
            'dist_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
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

find_unused_parameters=True
