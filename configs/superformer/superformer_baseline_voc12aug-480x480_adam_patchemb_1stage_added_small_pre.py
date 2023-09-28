_base_ = [
    './superformer_baseline_voc12aug-480x480.py',
]
model = dict(
    decode_head=dict(
    use_patch_embed = True,
    use_stem = False,
    sp_iter = 0,
    classification_feature = 'superpixel_bilinear',),)    

