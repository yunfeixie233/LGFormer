_base_ = [
    './pixel_extra_small_pre_p7p8p9p10_nols_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_final_attn = False,
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=0.6),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_1', use_sigmoid=False, loss_weight=0.1),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_2', use_sigmoid=False, loss_weight=0.1),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_3', use_sigmoid=False, loss_weight=0.1),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt_4', use_sigmoid=False, loss_weight=0.1),
            ] 
))
