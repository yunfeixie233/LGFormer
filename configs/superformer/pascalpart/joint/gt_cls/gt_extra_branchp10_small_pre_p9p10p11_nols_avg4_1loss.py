_base_ = [
    './gt_extra_branchp11_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    depths_obj = (None,2,-1,),
    )
)
