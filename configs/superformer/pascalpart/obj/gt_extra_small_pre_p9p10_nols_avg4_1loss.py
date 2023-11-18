_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_1loss.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,),()), 
    group_init_strides = (4,4,),
    group_init_kernel_sizes = (4,4,),
    group_layers = {0:64,1:64,},   
))
