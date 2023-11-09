_base_ = [
    './gt_extra_small_pre_p9_nols_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(3,8,),()), 
    group_init_strides = (3,3,),
    group_init_kernel_sizes = (3,3,),
    group_layers = {0:100,1:100,},    
    group_token_init_method = ('avgpool','from_feature',), 
    ungroup_enable = (False,True,)
))
