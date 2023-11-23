_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(5,6,7,8,),()),        
    group_layers = {0:64,1:64,2:64,3:64}, # group token num each layer 
    group_init_strides = (4,4,4,4,),# group token init strides
    group_init_kernel_sizes = (4,4,4,4,),# group token init kernel sizes
    group_token_init_method = ('avgpool','from_feature','from_feature','from_feature',), 
    group_identity = (True, True,True,True,), 
    ungroup_ls_init_value = None,
    use_ffn = True,
    ungroup_enable = (False,False,False,True)
))
