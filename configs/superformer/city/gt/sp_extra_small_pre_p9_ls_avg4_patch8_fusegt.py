_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_patch8.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        use_gt_fuse = True,
        gt_cls_method = 'upsample_first',
        use_gt_extralayer = True,        
))
