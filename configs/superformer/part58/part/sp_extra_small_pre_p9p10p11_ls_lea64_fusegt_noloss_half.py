_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss_half.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_token_init_method = ('learnable','learnable','learnable',),   
))