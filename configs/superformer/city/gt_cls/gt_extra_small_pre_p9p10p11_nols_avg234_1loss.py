_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg3_all.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(7,8,9,),()), 
    group_init_strides = (2,3,4,),
    group_init_kernel_sizes = (2,3,4,),
    group_layers = {0:576,1:256,2:144,},          
    use_final_group_cls = True,    
    group_token_init_method = ('avgpool','avgpool','avgpool',), 
                    
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
