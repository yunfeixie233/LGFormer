_base_ = [
    './sp_extra_small_pre.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_gt_extralayer = True,        
    use_gt_loss = True,
    group_pos = ((),(7,),()), 
    use_group_token = 'mix',
    group_init_strides = (4,),
    group_init_kernel_sizes = (4,),
    group_layers = {0:64,},
    ungroup_ls_init_value = None,
    use_ffn = True,
    use_final_group_cls = False,
    use_final_group = False,
    use_group_attn = True,      
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj',use_sigmoid=False, loss_weight=0.5),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part', use_sigmoid=False, loss_weight=0.5)]
         
))
