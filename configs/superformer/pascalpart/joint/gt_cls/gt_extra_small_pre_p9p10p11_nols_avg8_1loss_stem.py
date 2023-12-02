_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg8_1loss.py',
]
num_classes = 57
model = dict( 
    decode_head=dict(
    group_init_strides = ((2,2,2,),(2,2,2,),(2,2,2,),),
    group_init_kernel_sizes = ((2,2,2,),(2,2,2,),(2,2,2,),),
))
