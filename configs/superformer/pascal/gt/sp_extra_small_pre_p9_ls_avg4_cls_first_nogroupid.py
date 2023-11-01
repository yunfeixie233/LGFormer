_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    gt_cls_method = 'cls_first',
    use_gt_extralayer = True,
))
