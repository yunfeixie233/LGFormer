_base_ = [
    './gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py',
]
model = dict(
    decode_head=dict(
    dims=(768, 768, 768),
    heads=(12, 12, -1),
    sp_heads=(3, 3, 1),
    stem_channels_list=(96,),
    drop_path_rate = 0.1,     
    stem_kernel_sizes=(4,),
    stem_conv_types=("conv",),
    stem_strides=(4,),   
    group_embed_dims=768,
    
    )
)
