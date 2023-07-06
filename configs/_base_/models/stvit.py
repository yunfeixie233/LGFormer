# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='STViT',
       embed_dim=[256, 256, 256, 256],
                    depths=[3, 5, 9, 3],
                    num_heads=[2, 2, 4, 4],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 8, 8, 8],
                    projection=256,                    
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0,
                    drop_path_rate=0.3, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, False, False],
                    init_values=1e-5,
                    seg=False),
    decode_head=dict(
        type='SViTHead',
       stoken_size=[8, 8, 8, 8],
       in_channels=256,
       channels=256,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))