_base_ = [
    './pixel_extra_small_pre_p7p8p9p10_nols_avg4_fusesp_fusegt_ffn_1ungroup.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'group_extralayer'
))
