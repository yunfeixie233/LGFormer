_base_ = [
    './sp_extra_small_pre_p8_ls_avg4_loss_6h.py'
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    reweight_gt = "sigmoid",
    output_dir = '/data2/yunfei/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi_reweightgt_sigmoid_vis', 
    vis_sp_block = True,    
))
