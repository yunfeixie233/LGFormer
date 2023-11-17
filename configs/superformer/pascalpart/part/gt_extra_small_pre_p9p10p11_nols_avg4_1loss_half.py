_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg2_fusegt_noloss_half.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_init_strides = (4,4,4,),
    group_init_kernel_sizes = (4,4,4,),
    group_layers = {0:64,1:64,2:64},       
))
