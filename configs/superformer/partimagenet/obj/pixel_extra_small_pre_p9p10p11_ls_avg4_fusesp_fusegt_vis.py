_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt.py',
]

model = dict(                 
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'pixel_extralayer',
    extralayer_nols = True,
    use_sp_fuse = True,   
    output_dir = "/data2/yunfei/pixel_extra_small_pre_p9p10p11_ls_avg4_fusesp_fusegt_vis",
    vis_spgt = True,
    vis_sp_id = True, 
    vis_gt = True,                                               
))
