_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt_convstem.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    reweight_pixel_update = True,
    log_reweight = True,
         
))
