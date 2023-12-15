_base_ = [
    './gt_extra_branchp9_small_pre_p9p10p11_nols_avg4_1loss_convstem_nosharedblocks_sharedgroup.py',
]
model = dict(
    decode_head=dict(
    group_pos = [[],[7,8,],[]],    
    group_init_strides = (4,4,),
    group_init_kernel_sizes = (4,4,),
    group_layers = {0:64,1:64,},   
    group_token_init_method = ('avgpool','from_feature',),    

    )
)
