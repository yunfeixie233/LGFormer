_base_ = [
    './sp_extra_small_pre_p8_ls_avg4_loss_6h.py'
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    expand_gt = True,
    keep_multihead = True,
    output_dir = '/data2/yunfei/', 
    vis_spgt = True,   
))
