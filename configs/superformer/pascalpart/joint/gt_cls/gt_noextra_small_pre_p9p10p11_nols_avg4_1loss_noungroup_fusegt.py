_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    classification_feature = 'joint',
    resize_similarity =  True,          
    depths=(2,10,),
    dims=(384,384,),
    heads=(6,6,),
    strides=(1,1,),
    sp_sizes=(4,4,),
    sp_heads=(2,2,),
    sp_features_init_methods=("from_feature","from_feature",),        
    use_final_group_cls = True,
    ungroup_enable = (False,False,False,),
    use_gt_fuse = True,                         
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
