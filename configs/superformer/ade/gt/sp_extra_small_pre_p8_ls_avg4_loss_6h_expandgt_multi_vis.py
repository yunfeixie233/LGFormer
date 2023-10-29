_base_ = [
    './sp_extra_small_pre_p8_ls_avg4_loss_6h.py'
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    expand_gt = True,
    keep_multihead = True,
<<<<<<< HEAD
    output_dir = '/data2/yunfei/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi_vis_2', 
    vis_sp_block = True   
=======
    output_dir = '/data2/yunfei/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi_vis', 
    # vis_gt = True,
    # vis_gt_eff = True,
    vis_sp_block = True,
>>>>>>> 35743ae0b0e14e749e2d1916d8e62312856e42f2
))
