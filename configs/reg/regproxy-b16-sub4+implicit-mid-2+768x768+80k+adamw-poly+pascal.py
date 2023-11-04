_base_ = [
    '../_base_/models/regproxy/regproxy-b16.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/adamw+poly-power_1+lr_6e-5+wd_0.01+iter_80k.py'
]
model = dict(
    backbone=dict(
        img_size=(480, 480),
        out_indices=[2, 11]),
    test_cfg=dict(
        mode='slide',
        crop_size=(480, 480),
        stride=(312, 312)))