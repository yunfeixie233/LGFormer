_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt.py',
]
accumulative_counts = 2
total_iter=50000 * accumulative_counts
param_scheduler = [
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=total_iter,
        eta_min=0.0,
        by_epoch=False,
    )
]
