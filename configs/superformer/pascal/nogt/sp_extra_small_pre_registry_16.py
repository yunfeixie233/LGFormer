_base_ = [
    './sp_extra_small_pre.py', 
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_global_token = True,
    num_global_token = 16,
    
))