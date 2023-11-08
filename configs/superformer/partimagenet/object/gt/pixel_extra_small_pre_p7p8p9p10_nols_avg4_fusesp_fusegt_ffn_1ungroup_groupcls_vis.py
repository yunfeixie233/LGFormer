_base_ = [
    './pixel_extra_small_pre_p7p8p9p10_nols_avg4_fusesp_fusegt_ffn_1ungroup.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'group_extralayer',
    output_dir = "/data2/yunfei/pixel_extra_small_pre_p7p8p9p10_ls_avg4_fusesp_fusegt_vispixel_extra_small_pre_p7p8p9p10_nols_avg4_fusesp_fusegt_ffn_1ungroup_groupcls_vis",
    vis_gt = True,
    vis_spgt = True,
    vis_sp_id = True,    
))
