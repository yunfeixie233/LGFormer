_base_ = [
    './sp_extra_small_pre.py'
]
model = dict(
    decode_head=dict(
        reweight_pixel_update = True,
        vis_pixel = True,
        vis_sp = True,
        output_dir = '/data2/yunfei/sp_extra_small_pre_reweight_pfeat_vis'
))
