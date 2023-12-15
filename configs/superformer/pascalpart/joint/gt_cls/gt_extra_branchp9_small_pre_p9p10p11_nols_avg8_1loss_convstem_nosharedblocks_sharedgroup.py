_base_ = [
    './gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py',
]
model = dict(
    decode_head=dict(
    group_init_strides = (8,8,8,),
    group_init_kernel_sizes = (8,8,8,),
    group_layers = {0:16,1:256,2:256},      
    )
)
