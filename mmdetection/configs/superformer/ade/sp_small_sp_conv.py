
_base_ = [
    './sp_small.py'
     
]
crop_size = (512, 512)
num_classes = 100
norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    type='MaskRCNN',
    neck=dict(
        _delete_=True,
        type='SpFPN',
        pixel_channels=64,
        sp_channels=384,
        out_channels=256,
        downsample_method = 'conv',        
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            num_classes = num_classes,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

