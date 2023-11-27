_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_final_group_cls = True,
    group_pos = ((),(0,1,2,),()),
    ungroup_enable = (False,False,False,),
    use_gt_fuse = True,                         
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
