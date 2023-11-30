_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_fuseloss_fuseconv_ft.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    classification_feature = 'joint_extralayer',       
    group_pos = ((),(5,6,7,),()), 

))
