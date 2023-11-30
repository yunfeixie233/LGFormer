_base_ = [
    './sp_extra_small_pre_p5p6p7_ls_avg4_fuseloss_fuseconv_fusegt.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    classification_feature = 'joint_extralayer',       
    use_gt_extralayer = False,     
))
