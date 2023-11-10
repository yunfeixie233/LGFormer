_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_groupattn_allloss_reinit.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_final_group_cls = False,
))
