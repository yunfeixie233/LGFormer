_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_fuseloss_fuseconv_loss.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    output_dir = "/data2/yunfei/sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_fuseloss_fuseconv_loss",
    vis_gt = True,
    vis_gt_eff = True,
    vis_sp = True,
    vis_spgt = True,
    vis_sp_id = True,          
))
