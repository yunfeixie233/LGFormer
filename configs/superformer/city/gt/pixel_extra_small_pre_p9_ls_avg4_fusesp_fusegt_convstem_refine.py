_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt_convstem.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    pixel_refine_method = 'sep_conv_3'
         
))
