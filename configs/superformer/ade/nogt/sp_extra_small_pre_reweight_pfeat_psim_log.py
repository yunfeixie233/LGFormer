_base_ = [
    './sp_extra_small_pre.py'
]
model = dict(
    decode_head=dict(
        reweight_pixel_update = True,
        reweight_pixel_sim = True,        
        log_reweight = True,
))
