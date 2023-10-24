_base_ =[
    './sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi.py'
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    reweight_pixel_update = True,
    log_reweight = True,    
))
