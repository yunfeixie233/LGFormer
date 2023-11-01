_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_sp_fuse = True,
))
