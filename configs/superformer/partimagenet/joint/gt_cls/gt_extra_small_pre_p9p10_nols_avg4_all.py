_base_ = [
    './gt_extra_small_pre_p9_nols_avg4_all.py',
]
model = dict(
    
    decode_head=dict(
    group_pos = ((),(7,8,),()), 
    group_init_strides = (4,4,),
    group_init_kernel_sizes = (4,4,),
    group_layers = {0:64,1:64}, 
    group_token_init_method = ('avgpool','avgpool',), 
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=0.8),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_1', use_sigmoid=False, loss_weight=0.1),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_2', use_sigmoid=False, loss_weight=0.1),
            ],     
))
