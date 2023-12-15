_base_ = [
    './gt_extra_branchp3_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    depths_obj = (None,4,-1,),
    group_pos = [[],[7,8,9,],[]],    
    onlyobj_merge_layer = False,
    part_cls_method = "cls_first",
    stem_kernel_sizes=(3, 3),
    stem_conv_types=("conv", "conv"),
    stem_channels_list=(32, 64),
    stem_strides=(2, 2),   
    shared_blocks = False,     
    shared_merge_layer = True,     

    )
)
