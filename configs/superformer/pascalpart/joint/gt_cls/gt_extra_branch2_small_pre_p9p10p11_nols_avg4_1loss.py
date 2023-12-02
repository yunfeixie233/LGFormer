_base_ = [
    './gt_extra_branch1_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    obj_stages_pos = (0,1,2,),
    )
)
