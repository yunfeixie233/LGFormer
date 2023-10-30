_base_ = [
    '../nogt/sp_extra_small_pre.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    ls_init_value = 1e-5,
    use_group_token = 'mix',
    use_gt_loss = True,
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=0.5),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp', use_sigmoid=False, loss_weight=0.5)]
    
))
