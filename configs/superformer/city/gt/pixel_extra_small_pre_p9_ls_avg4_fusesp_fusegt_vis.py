_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    output_dir = "/data2/yunfei/pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt_vis",
    vis_gt = True,
    vis_sp = True,
    vis_spgt = True,
    vis_sp_id = True,
    group_identity = False,
    group_pe_method = None,
))
