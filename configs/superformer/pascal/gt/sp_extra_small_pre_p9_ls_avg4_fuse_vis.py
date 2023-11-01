_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_fuse.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        output_dir = '/data2/yunfei/sp_extra_small_pre_p9_ls_avg4_fuse'
))
