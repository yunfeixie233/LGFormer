_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_groupattn_allloss_reinit_allcls_new.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_init_strides = (2,3,5,),
    group_init_kernel_sizes = (2,3,5,),
    group_layers = {0:225,1:100,2:36,},        
))
