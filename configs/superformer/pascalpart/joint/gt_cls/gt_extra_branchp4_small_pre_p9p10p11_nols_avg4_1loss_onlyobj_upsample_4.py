_base_ = [
    './gt_extra_branchp3_small_pre_p9p10p11_nols_avg4_1loss.py',
]
model = dict(
    decode_head=dict(
    depths_obj = (None,8,-1,),
    onlyobj_merge_layer = True,
    part_cls_method = "upsample_first",
    )
)
