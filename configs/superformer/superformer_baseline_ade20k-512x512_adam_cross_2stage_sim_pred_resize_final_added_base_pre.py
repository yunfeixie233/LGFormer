_base_ = [
    "./superformer_baseline_ade20k-512x512_adam_cross_2stage_sim_pred_resize_final_added_base.py"
]
model = dict(
    decode_head=dict(
        ls_init_value=1e-5,
    ),
)
