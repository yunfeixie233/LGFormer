_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    
    decode_head=dict(
    use_final_group_cls = True,
    group_pos = ((),(3,4,5,),()),                         
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_sp',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],
))
accumulative_counts = 2
total_iter=50000 * accumulative_counts
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500* accumulative_counts),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500* accumulative_counts,
        end=total_iter,
        eta_min=0.0,
        by_epoch=False,
    )
]