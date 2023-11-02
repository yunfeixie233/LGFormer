_base_ = [
    './sp_extra_small_pre_p9p10_ls_avg4_fromfeat_id.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(6,7,),()),        

))
