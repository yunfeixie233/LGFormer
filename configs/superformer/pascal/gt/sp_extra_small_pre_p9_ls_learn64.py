_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_token_init_method = 'learnable',
    group_layers = {0:64,}, # group token num each layer 
        
))
