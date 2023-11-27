_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    
    decode_head=dict(
    use_final_group_cls = True,
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_gt',use_sigmoid=False, loss_weight=1.0,reduction='mean',),]
))
accumulative_counts = 4
total_iter=10000 * accumulative_counts
optim_wrapper = dict(
    _delete_ = True,
    type='OptimWrapper',
    optimizer=dict(
    type='AdamW',
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.05),
    accumulative_counts=accumulative_counts,    
    paramwise_cfg=dict(
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
            'merge_layer':dict(lr_mult=10.),        
        }))