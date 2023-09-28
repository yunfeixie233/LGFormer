_base_ = ["./superformer_baseline_ade20k-512x512_adam_patchemb_1stage_added.py"]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        dims=(768,),
        heads=(12,),
        sp_heads=(3,),
        stem_channels_list=(96,),
    ),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(512, 512)),
)
