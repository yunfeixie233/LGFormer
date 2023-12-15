_base_ = [
    './gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py',
]
optim_wrapper = dict(
clip_grad=dict(max_norm=10, norm_type=2))