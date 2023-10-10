_base_ = [
    './superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre.py',
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
        'num_layers': 2,
        'num_heads': 2,
        'num_group_heads': 2,
        'num_group_forward_heads': 2,
        'num_ungroup_heads': 2,
        'ffn_ratio': 4.,
        'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),
        'mlpmixer_depth': 1,
        'group_layers': {0:100,1:25,},
        'drop_path_rate': 0.2,
        'group_projector_methonds':'linear',
        'association_embedding':False,
        'group_token_init_method':'conv',
        "init_strides":(3,6,),
        'init_kernel_sizes':(3,6,),
        'merge_pos':((),(6,8,),()),
        'ls_init_value':0.,            
    },
    
),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)))

