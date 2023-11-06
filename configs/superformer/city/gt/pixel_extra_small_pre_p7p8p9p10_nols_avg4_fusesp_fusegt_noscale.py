_base_ = [
    './pixel_extra_small_pre_p7p8p9p10_nols_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_qk_scale = 1,
))
