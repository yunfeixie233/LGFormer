_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss_half.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (2,2,2,),
    group_init_kernel_sizes = (2,2,2,),
    group_layers = {0:256,1:256,2:256},          
    group_token_init_method = ('conv','conv','conv',),   
))