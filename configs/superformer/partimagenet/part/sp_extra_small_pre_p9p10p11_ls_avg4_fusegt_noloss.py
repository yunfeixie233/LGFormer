_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_gt_loss = False,
    use_final_group = True,
    loss_decode=[
        dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),],    

))
