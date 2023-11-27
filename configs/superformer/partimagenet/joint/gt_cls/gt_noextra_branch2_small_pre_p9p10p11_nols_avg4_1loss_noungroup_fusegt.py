_base_ = [
    './gt_noextra_small_pre_p9p10p11_nols_avg4_1loss_noungroup_fusegt.py',
]
model = dict(
    decode_head=dict(
    group_stages_pos = (0,1,),
))
