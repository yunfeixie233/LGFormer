_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_fuseloss_fuseconv_ft_move2.py',
]
num_classes = 57
accumulative_counts = 4
total_iter=10000 * accumulative_counts
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=total_iter,
        eta_min=0.0,
        by_epoch=False,
    )
]
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
        }))
