_base_ = [
    './sp_extra_small_pre_p9_ls_avg4_nogroupid.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    use_global_token = True,
    num_global_token = 16,    
    use_global_fuse = False,        
))
