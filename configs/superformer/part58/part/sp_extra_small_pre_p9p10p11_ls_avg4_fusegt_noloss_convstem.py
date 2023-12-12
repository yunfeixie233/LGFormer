_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss.py',
]

model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    stem_kernel_sizes=(3, 3),
    stem_conv_types=("conv", "conv"),
    stem_channels_list=(32, 64),
    stem_strides=(2, 2),       

))
