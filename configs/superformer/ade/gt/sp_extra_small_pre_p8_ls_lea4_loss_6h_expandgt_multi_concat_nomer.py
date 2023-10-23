_base_ = [
    '../nogt/sp_extra_small_pre.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    ls_init_value = 1e-5,
    resize_version = 'v2',
    use_group_token = 'mix',
    arch_settings = {
        'embed_dims': 384,
        'patch_size': 8,
        'window_size': 2,
        'num_layers': 1,
        'num_heads': 6,
        'num_group_heads': 6,
        'num_group_forward_heads': 6,
        'num_ungroup_heads': 6,
        'ffn_ratio': 4.,
        'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),
        'mlpmixer_depth': 1,
        'group_layers': {0:16,},
        'drop_path_rate': 0.2,
        'group_projector_methonds':None,
        'association_embedding':False,
        'group_token_init_method':'learnable',
        "init_strides":(4,),
        'init_kernel_sizes':(4,),
        'merge_pos':((),(-1,),()),
        'ls_init_value':1e-5,            
    },
    use_gt_concat = True,
))
