_base_ = [
    './gt_extra_small_pre_p9p10_nols_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_drop_path_rate=0.1,
))
