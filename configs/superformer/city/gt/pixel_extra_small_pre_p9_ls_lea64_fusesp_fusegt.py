_base_ = [
    '../nogt/sp_extra_small_pre.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_token_init_method = 'learnable',
    group_layers = {0:64,},
            
))
