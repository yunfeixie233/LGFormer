_base_ = [
    './pixel_extra_small_pre.py', 
]
model = dict(
    decode_head=dict(
        reweight_pixel_update = True,
        log_reweight = True,
))

