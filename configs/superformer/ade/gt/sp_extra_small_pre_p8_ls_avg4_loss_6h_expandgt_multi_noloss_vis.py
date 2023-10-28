_base_ = [
    './sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi_noloss.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    vis_gt = True,
    vis_spgt = True,
    output_dir = '/data2/yunfei/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi_noloss'
    
))
