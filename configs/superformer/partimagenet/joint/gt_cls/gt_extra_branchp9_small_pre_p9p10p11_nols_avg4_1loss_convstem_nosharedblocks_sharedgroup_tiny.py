_base_ = [
    './gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py',
]
model = dict(
    decode_head=dict(
        sp_heads=(1, 1, 1),
        stem_channels_list=(32,),
        stem_strides=(4,),   
        stem_kernel_sizes=(4,),        
        stem_conv_types=("conv",),           
        dims=(192, 192,192),
        heads=(3, 3, -1),        
        group_embed_dims=192,
        drop_path_rate=0.1,
        
    )
)
