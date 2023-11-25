_base_ = [
    './sp_extra_small_pre_p9p10p11_ls_avg4_fusegt.py',
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    drop_path_rate = 0.2,                                      
    
))
