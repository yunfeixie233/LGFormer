_base_ = [
    './sp_extra_small_pre.py', 
]
model = dict(
    decode_head=dict(
    classification_feature = 'regproxy',    
    depths=(2,10,),
    dims=(384,384,),
    heads=(6,6,),
    strides=(1,1,),
    sp_sizes=(4,4,),
    sp_heads=(2,2,),
    sp_features_init_methods=("from_feature","from_feature",),
))

