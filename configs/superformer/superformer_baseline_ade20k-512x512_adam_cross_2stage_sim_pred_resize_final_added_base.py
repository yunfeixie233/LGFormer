_base_ = [
    "./superformer_baseline_ade20k-512x512_adam_cross_2stage_sim_pred_resize_final_added.py"
]
model = dict(
    decode_head=dict(
        dims=(768, 768, 768),
        heads=(12, 12, -1),
        sp_heads=(3, 3, 1),
        stem_channels_list=(96,),
    ),
)
