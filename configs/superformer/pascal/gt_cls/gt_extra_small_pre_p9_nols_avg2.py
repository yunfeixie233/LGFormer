_base_ = [
    './gt_extra_small_pre_p9_nols_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict( 
    group_init_strides = (2,),
    group_init_kernel_sizes = (2,),
    group_layers = {0:225,},  
    use_group_attn = True,
  
))
