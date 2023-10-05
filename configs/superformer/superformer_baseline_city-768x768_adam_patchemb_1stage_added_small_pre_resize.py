_base_ = [
    './superformer_baseline_city-768x768.py',
]
model = dict(
    decode_head=dict(
    use_patch_embed = True,
    use_stem = False,
    sp_iter = 0,
    classification_feature = 'superpixel_bilinear',
    cls_scale_factor = 16,),)    

find_unused_parameters=False