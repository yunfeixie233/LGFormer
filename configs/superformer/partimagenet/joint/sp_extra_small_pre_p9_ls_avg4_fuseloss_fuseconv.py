_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_fusegt.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'joint_extralayer',       
    group_pos = ((),(7,),()), 
    group_init_strides = (4,),
    group_init_kernel_sizes = (4,),
    group_layers = {0:64,}, 
    use_final_group = True,
      
    group_token_init_method = ('avgpool',), 
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt', use_sigmoid=False, loss_weight=0.5,reduction='mean',),         
            ],         
))
