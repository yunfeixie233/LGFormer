_base_ = [
    './sp_extra_base_pre_p9_ls_learn8.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_group_token = 'mix',
    use_gt_loss = True,
    group_init_strides = (3,),
    group_init_kernel_sizes = (3,),
    group_layers = {0:100,},
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=0.5),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp', use_sigmoid=False, loss_weight=0.5)],
    group_pos = ((),(7,),()),
    group_identity =False,
))
