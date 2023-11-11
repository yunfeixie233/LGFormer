_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]
