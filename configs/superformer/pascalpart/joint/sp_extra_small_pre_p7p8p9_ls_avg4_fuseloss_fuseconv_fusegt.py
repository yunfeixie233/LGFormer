_base_ = [
    './sp_extra_small_pre_p7p8p9_ls_avg4_fuseloss_fuseconv.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    use_gt_fuse = True,
))
