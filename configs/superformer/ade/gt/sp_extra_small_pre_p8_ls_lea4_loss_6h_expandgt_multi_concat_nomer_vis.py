_base_ = [
    './sp_extra_small_pre_p8_ls_lea4_loss_6h_expandgt_multi_concat_nomer.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    vis_sp_stage = True,
    output_dir = '/data2/yunfei/sp_extra_small_pre_p8_ls_lea4_loss_6h_expandgt_multi_concat_nomer'
))
