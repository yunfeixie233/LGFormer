_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_token_init_method = 'depthwise',
    group_init_strides = (3,),# group token init strides
    group_init_kernel_sizes = (7,),# group token init kernel sizes          
))
