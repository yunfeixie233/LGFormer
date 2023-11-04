_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_patch8_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    output_dir = "/root/autodl-tmp/sp_extra_small_pre_p9_ls_avg4_patch8_fusegt_vis",
    vis_gt = True,
    vis_sp = True,
    vis_spgt = True,
    vis_sp_id = True,
      
))
