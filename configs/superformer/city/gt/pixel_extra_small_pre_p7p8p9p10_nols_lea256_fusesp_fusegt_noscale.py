_base_ = [
    './pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(5,6,7,8,),()),        
    group_qk_scale = 1,
))
