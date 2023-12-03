_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    depths_obj = (None,1,-1,),
    group_init_strides = (2,2,2,),
    group_init_kernel_sizes = (2,2,2,),
    group_layers = {0:256,1:256,2:256},      
    )
)
