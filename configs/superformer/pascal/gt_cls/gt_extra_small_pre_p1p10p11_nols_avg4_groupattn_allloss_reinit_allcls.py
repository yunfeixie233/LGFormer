_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_groupattn_allloss_reinit_allcls_new.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((1,),(8,9,),()), 
))
