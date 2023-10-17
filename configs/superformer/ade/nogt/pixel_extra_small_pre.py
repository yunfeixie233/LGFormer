_base_ = [
    './sp_extra_small_pre.py', 
]
model = dict(
    decode_head=dict(
    classification_feature = 'pixel_extralayer',
    extralayer_nols = True         
))

