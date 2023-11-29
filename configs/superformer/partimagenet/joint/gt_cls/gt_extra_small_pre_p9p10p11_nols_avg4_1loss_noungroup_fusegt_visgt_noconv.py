_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    use_final_group_cls = True,
    ungroup_enable = (False,False,False,),
    use_gt_fuse = True,     
    output_dir = "/data1/yunfei/vis_gt_noconv",                    
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
