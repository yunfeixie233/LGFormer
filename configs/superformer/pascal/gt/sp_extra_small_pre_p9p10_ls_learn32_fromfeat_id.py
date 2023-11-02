_base_ = [
    './sp_extra_small_pre_p9p10_ls_avg4_fromfeat_noid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_identity = (False,True),
    group_layers = {0:32,1:32,}, # group token num each layer 
    group_token_init_method=('learnable','from_feature',)

))
