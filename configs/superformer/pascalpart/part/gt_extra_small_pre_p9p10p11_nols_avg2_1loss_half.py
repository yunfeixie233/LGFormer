_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg2_fusegt_noloss_half.py',
]

model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'group_extralayer', 
    use_final_group_cls = True,
    use_gt_loss = True,
))
