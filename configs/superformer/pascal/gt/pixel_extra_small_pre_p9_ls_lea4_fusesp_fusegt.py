_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_gt_fuse = True,
    group_token_init_method = 'learnable' 
))
