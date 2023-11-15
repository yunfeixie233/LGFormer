_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss_half.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (8,8,8,),
    group_init_kernel_sizes = (8,8,8,),
    group_layers = {0:16,1:16,2:16},   
))