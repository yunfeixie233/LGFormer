_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,),()),        
    group_layers = {0:100,1:100,}, # group token num each layer 
    group_init_strides = (3,3,),# group token init strides
    group_init_kernel_sizes = (3,3,),# group token init kernel sizes
    group_identity =False,
    group_token_init_method=('avgpool','from_feature',)
))
