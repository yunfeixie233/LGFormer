_base_ = [
    './sp_extra_small_pre.py', 
]
model = dict(
    decode_head=dict(
        dims=(768, 768, 768),
        heads=(12, 12, -1),
        sp_heads=(3, 3, 1),
        stem_channels_list=(96,),
             
))

