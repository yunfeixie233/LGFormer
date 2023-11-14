_base_ = [
    './sp_extra_small_pre_p7p8p9_ls_avg4_fusegt_fuseloss_fuseconv_loss.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    ungroup_enable = False,
))
