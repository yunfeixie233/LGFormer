_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(5,6,7,8,),()),        
    group_layers = {0:256,1:256,2:256,3:256}, # group token num each layer 
    group_init_strides = (3,3,3,3,),# group token init strides
    group_init_kernel_sizes = (3,3,3,3,),# group token init kernel sizes
    group_token_init_method = ('avgpool','avgpool','avgpool','avgpool',), 
    group_identity = (True, True,True,True,)  
))
