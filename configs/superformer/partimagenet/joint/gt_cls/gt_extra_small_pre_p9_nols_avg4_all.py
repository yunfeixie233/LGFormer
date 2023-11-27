_base_ = [
    '../gt/pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    
    decode_head=dict(
    classification_feature = 'joint_extralayer',
    ungroup_ls_init_value = None,
    use_ffn = True,
    use_final_group_cls = False,
    use_final_group = False,
    use_group_attn = True,    
))
