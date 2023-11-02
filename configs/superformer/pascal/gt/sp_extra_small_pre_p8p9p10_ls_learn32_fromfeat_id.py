_base_ = [
    './sp_extra_small_pre_p9p10_ls_avg4_fromfeat_noid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(6,7,8,),()),          
    group_identity = (False,True,True),
    group_init_strides = (0,0,0,),# group token init strides
    group_init_kernel_sizes = (0,0,0,),# group token init kernel sizes
    
    group_layers = {0:32,1:32,2:32,}, # group token num each layer 
    group_token_init_method=('learnable','from_feature','from_feature',)

))
