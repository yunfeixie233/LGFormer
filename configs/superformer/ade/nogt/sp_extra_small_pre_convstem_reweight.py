_base_ = [
    './sp_extra_small_pre_convstem.py'
]
model = dict(
    decode_head=dict(
        reweight_pixel_update = True,
        log_reweight = True,        
))
