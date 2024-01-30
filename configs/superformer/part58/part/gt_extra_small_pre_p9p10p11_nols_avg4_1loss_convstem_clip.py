_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_1loss_convstem.py',
]

optim_wrapper = dict(
clip_grad=dict(max_norm=20, norm_type=2)
)
