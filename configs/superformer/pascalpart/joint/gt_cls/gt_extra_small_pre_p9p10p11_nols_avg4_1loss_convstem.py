_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_1loss.py',
]
num_classes = 57
model = dict(
    
    decode_head=dict(
    stem_kernel_sizes=(3, 3),
    stem_conv_types=("conv", "conv"),
    stem_channels_list=(32, 64),
    stem_strides=(2, 2),))
