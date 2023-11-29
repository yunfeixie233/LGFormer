_base_ = [
    './gt_extra_small_pre_p9_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (4,4,4,),
    group_init_kernel_sizes = (4,4,4,),
    group_layers = {0:64,1:64,2:64},   
    group_token_init_method = ('avgpool','from_feature','from_feature',), 
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part',use_sigmoid=False, loss_weight=0.7,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj_1', use_sigmoid=False, loss_weight=0.1,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj_2', use_sigmoid=False, loss_weight=0.1,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj_3', use_sigmoid=False, loss_weight=0.1,reduction='mean',),            
            ],         
))
