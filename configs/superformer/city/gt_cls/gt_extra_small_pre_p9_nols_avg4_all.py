_base_ = [
    '../gt/pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'group_extralayer',
    ungroup_ls_init_value = None,
    use_ffn = True,
    use_final_group_cls = False,
    use_final_group = False,
    use_group_attn = True,    
))
