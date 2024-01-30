_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_1loss_convstem.py',
]

model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    freeze_embedding = True,
    ))
