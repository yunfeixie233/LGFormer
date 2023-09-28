_base_ = [
    './superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        dims=(768, 768, 768),
        heads=(12, 12, -1),
        sp_heads=(3, 3, 1),
        stem_channels_list=(96,),)),



