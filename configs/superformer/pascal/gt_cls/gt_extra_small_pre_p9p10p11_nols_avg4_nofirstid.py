_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_identity =(False,True,True,)
))
