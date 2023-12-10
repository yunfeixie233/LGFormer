_base_ = [
    './gt_extra_branchp11_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    depths_obj = (None,9,-1,),
    )
)
accumulative_counts = 4
total_iter=10000 * accumulative_counts
optim_wrapper = dict(accumulative_counts=accumulative_counts)    

param_scheduler = [
    dict(
            type='MultiStepLR',
            begin=0,                     # 从第0个epoch开始
            end=total_iter,            # 在总训练周期结束时停止更新学习率
            by_epoch=False,               # 通过epoch来更新学习率
            milestones=[int(total_iter * 0.9), int(total_iter * 0.95)],  # 在第90个和第95个epoch降低学习率
            gamma=0.1,                   # 学习率衰减因子
            verbose=False                # 设置为True以打印每次更新的学习率
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iter, val_interval=1000)
train_dataloader = dict(
    batch_size=4)