_base_ = [
    './pixel_extra_base_pre_p9_ls_avg4_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    drop_path_rate = 0.1    
         
))
