_base_ = [
    './sp_extra_small_pre_p9_ls_learn32_convstem.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_init_strides = (4,),
    group_init_kernel_sizes = (4,),
  
        
))
