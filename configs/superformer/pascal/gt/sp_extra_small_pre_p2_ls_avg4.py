_base_ = [
    './sp_extra_small_pre_p10_ls_avg4.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        group_pos = ((),(0,),()),
    
))
