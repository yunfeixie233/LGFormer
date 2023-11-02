_base_ = [
    './sp_extra_small_pre_p9p10_ls_avg4_fromfeat_id.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(    
    group_layers = {0:100,1:36,}, # group token num each layer 
    group_init_strides = (3,5,),# group token init strides
    group_init_kernel_sizes = (3,5,),# group token init kernel sizes

))
