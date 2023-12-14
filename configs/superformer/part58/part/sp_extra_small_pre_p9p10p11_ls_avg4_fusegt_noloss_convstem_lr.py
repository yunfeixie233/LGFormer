_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt_noloss.py',
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
    type='SGD',
    lr=0.007,
    momentum=0.9,
    weight_decay=1e-4,),)
