_base_ = [
    './sp_extra_small_pre.py', 
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
    vis_pixel = True,
    vis_sp = True,
    output_dir = '/data2/yunfei/sp_extra_small_pre_vis'))

