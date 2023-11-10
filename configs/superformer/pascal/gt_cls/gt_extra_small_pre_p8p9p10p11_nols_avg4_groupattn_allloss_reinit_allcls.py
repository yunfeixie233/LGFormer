_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_groupattn_allloss_reinit_allcls_new.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(6,7,8,9,),()), 
    group_init_strides = (3,3,3,3,),
    group_init_kernel_sizes = (3,3,3,3,),
    group_layers = {0:100,1:100,2:100,3:100,},    
    group_token_init_method = ('avgpool','avgpool','avgpool','avgpool'), 
))
