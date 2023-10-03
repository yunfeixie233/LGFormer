_base_ = [
    './superformer_baseline_city-768x768.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    classification_feature = 'superpixel_extralayer',
    resize_similarity =  True,          
    depths=(2,10,0,),
    dims=(384,384,384,),
    heads=(6,6,-1,),
    strides=(1,1,1,),
    sp_sizes=(4,4,4,),
    sp_heads=(2,2,1,),
    ls_init_value = 1e-5,    
    sp_features_init_methods=("from_feature","from_feature","from_feature",),))



