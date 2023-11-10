_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_groupattn_allloss.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=0.55),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_1', use_sigmoid=False, loss_weight=0.15),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_2', use_sigmoid=False, loss_weight=0.15),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_3', use_sigmoid=False, loss_weight=0.15),            
            ],          
))
