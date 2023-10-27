_base_ = [
    './sp_extra_small_pre_p8_ls_avg4_loss_6h.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'group_extralayer',   
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
        'group_layers': {0:64,},
        'drop_path_rate': 0.2,
        'group_projector_methonds':'linear',
        'association_embedding':False,
        'group_token_init_method':'avgpool',
        "init_strides":(4,),
        'init_kernel_sizes':(4,),
        'merge_pos':((),(6,),()),
        'ls_init_value':1e-5,            
    },     
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0)],
    use_gt_loss = False,
    expand_gt = True,
    keep_multihead = True,
))
