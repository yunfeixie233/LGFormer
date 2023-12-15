_base_ = [
    './gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py',
]
model = dict(
    decode_head=dict(
    group_init_strides = (2,2,2,),
    group_init_kernel_sizes = (2,2,2,),
    group_layers = {0:256,1:256,2:256},      
    )
)
