_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_layers = {0:225,}, # group token num each layer 
    group_init_strides = (2,),# group token init strides
    group_init_kernel_sizes = (2,),# group token init kernel sizes        
))
