_base_ = [
    './gt_extra_small_pre_p9_nols_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_ls_init_value = 1e-5
    
))
