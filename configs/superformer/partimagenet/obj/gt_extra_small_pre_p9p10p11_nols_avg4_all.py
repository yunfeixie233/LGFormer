_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'group_extralayer', 
))
