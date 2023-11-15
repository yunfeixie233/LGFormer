_base_ = [
    './gt_extra_small_pre_p9_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (4,4,4,),
    group_init_kernel_sizes = (4,4,4,),
    group_layers = {0:64,1:64,2:64},   
    group_token_init_method = ('avgpool','avgpool','avgpool',), 
    use_final_group_cls = True,    
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
