_base_ = [
    './sp_extra_small_pre.py', 
]
model = dict(
    decode_head=dict(
    classification_feature = 'superpixel_extralayer_similarity',
    resize_similarity =  True,          
))

