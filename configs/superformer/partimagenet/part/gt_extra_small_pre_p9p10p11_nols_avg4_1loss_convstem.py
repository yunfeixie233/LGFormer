_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    stem_kernel_sizes=(3, 3),
    stem_conv_types=("conv", "conv"),
    stem_channels_list=(32, 64),
    stem_strides=(2, 2),            
    use_final_group_cls = True,    
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
