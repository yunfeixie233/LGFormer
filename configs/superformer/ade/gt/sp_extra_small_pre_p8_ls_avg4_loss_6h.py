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
        'embed_dims': 384, #group token dim
        'patch_size': 8, #deprecated
        'window_size': 2, #deprecated
        'num_layers': 1, #group token layer num
        'num_heads': 6, #group token head num
        'num_group_heads': 6, # cross attention head num
        'num_group_forward_heads': 6,#deprecated
        'num_ungroup_heads': 6,# cross attention head num
        'ffn_ratio': 4.,
        'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),#deprecated
        'mlpmixer_depth': 1, #self attention block num
        'group_layers': {0:64,}, # group token num each layer
        'drop_path_rate': 0.2, # deprecated
        'group_projector_methonds':'linear', # projection method if we have token from prev layer
        'association_embedding':False, # deprecated
        'group_token_init_method':'avgpool', 
        "init_strides":(4,), # group token init strides
        'init_kernel_sizes':(4,),# group token init kernel sizes
        'merge_pos':((),(8,),()), # where to use group token ((),(8,),()) means we do not use group token at stage 1 and stage 3, use group after 8th vit block in stage 2
        'ls_init_value':1e-5,     # ls in group       
    },
    use_gt_loss = True,
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=0.5),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp', use_sigmoid=False, loss_weight=0.5)]
    
))
