_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt_patch8.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    output_dir = "/data2/yunfei/pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt_patch8",
    vis_gt = True,
    vis_sp = True,
    vis_spgt = True,
    vis_sp_id = True,
               
))
