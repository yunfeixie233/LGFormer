_base_ = [
    "./gt_cls/gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py",
]

model = dict(
    decode_head=dict(
    depths_obj = None,
    group_pos =None,    
    use_gt_loss = False,
    classification_feature = 'joint_superpixel',
    stem_kernel_sizes=(3, 3),
    stem_conv_types=("conv", "conv"),
    stem_channels_list=(32, 64),
    stem_strides=(2, 2),   
    use_group_token = False,
    )
)
