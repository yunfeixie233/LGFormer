_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
model = dict(
    type='EncoderDecoderJoint',    
    decode_head=dict(
    obj_stages_pos = (0,1,2,),
    classification_feature = 'joint_extralayer',
    use_gt_fuse = False,
    use_final_group_cls = True, 
    ungroup_enable = (False,False,False,),   
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part',use_sigmoid=False, loss_weight=1.0,reduction='mean',),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj',use_sigmoid=False, loss_weight=1.0,reduction='mean',)],    
))

accumulative_counts = 4
total_iter=50000 * accumulative_counts
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05),
    accumulative_counts=accumulative_counts,    
    paramwise_cfg=dict(
        bypass_duplicate=True,        
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm1': dict(decay_mult=0.),
            'norm2': dict(decay_mult=0.),
            'ls':dict(decay_mult=0.),
            'ln':dict(decay_mult=0.),
            'bn':dict(decay_mult=0.),
            'stem.conv_layers.0.1':dict(decay_mult=0.),
            'sp_project.1':dict(decay_mult=0.),
            'sp_init.1':dict(decay_mult=0.),
            'seg_norm': dict(decay_mult=0.),
            'gamma': dict(decay_mult=0.),
            'reweight': dict(decay_mult=0.),
            'stem':dict(lr_mult=0.1),
            'sp_init':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.0':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1._sp_qkv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1._pixel_qkv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.sp_pos_conv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.pixel_pos_conv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.sp_ls1.':dict(lr_mult=0.1),
            'stages.0.sp_project':dict(lr_mult=0.1),
            'stages.0.blocks':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.0':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1._sp_qkv.':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1._pixel_qkv.':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1.sp_pos_conv.':dict(lr_mult=0.1),
            'stages.1.patch_embed.blocks.1.pixel_pos_conv.':dict(lr_mult=0.1),
            'stages.0.patch_embed.blocks.1.sp_ls1.':dict(lr_mult=0.1),         
            'stages.1.blocks':dict(lr_mult=0.1)},    
        ))

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