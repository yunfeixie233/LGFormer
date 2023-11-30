_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_fusegt.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    classification_feature = 'joint_extralayer',       
    group_pos = ((),(3,4,5,),()), 
    group_init_strides = (4,4,4,),
    group_init_kernel_sizes = (4,4,4,),
    group_layers = {0:64,1:64,2:64}, 
    use_final_group = True,
    use_gt_fuse = True,      
    group_token_init_method = ('avgpool','avgpool','avgpool',), 
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj', use_sigmoid=False, loss_weight=0.1,reduction='mean',),         
            ],         
))
accumulative_counts = 4
total_iter=50000 * accumulative_counts
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500* accumulative_counts),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500* accumulative_counts,
        end=total_iter,
        eta_min=0.0,
        by_epoch=False,
    )
]
