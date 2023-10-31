_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_block_depth =3,
        
))
