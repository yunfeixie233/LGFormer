_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,),()),        
    group_layers = {0:256,1:256,}, # group token num each layer 
    group_init_strides = (3,3,),# group token init strides
    group_init_kernel_sizes = (3,3,),# group token init kernel sizes
    group_token_init_method = ('avgpool','from_feature',), 
    group_identity = (True, True,),
    group_ls_init_value = 1e-5,  
))
