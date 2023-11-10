_base_ = [
    './gt_extra_small_pre_p9p10_nols_avg4_groupattn_allloss.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_token_init_method = ('avgpool','avgpool',),     
))
