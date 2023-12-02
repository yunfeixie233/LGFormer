_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_1loss.py',
]
num_classes = 57
model = dict(
    
    decode_head=dict(
    group_token_init_method = ('learnable','from_feature','from_feature',),  ))
