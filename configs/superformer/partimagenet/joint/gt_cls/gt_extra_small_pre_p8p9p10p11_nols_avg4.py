_base_ = [
    './gt_extra_small_pre_p9_nols_avg4_all.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(6,7,8,9,),()), 
    group_init_strides = (4,4,4,4,),
    group_init_kernel_sizes = (4,4,4,4,),
    group_layers = {0:64,1:64,2:64,3:64},    
    group_token_init_method = ('avgpool','avgpool','avgpool','avgpool',), 
    
))
