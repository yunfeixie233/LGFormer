# dataset settings
dataset_type = 'ADE20KInstanceDataset'
data_root = '/data2/yunfei/SpformerV1/data/ade/ADEChallengeData2016/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/ADEChallengeData2016/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None
crop_size = (512,512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ade20k_instance_train.json',
        data_prefix=dict(img='images/training'),
        test_mode=False,
        pipeline=train_pipeline,
        backend_args=backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=False), 
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),

    # If you don't have a gt annotation, delete the pipeline    
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ade20k_instance_val.json',
        data_prefix=dict(img='images/validation'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='ade20k_instance_train.json',
#         data_prefix=dict(img='images/training'),
#         test_mode=False,
#         pipeline=train_pipeline,
#         backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'ade20k_instance_val.json',
    # ann_file=data_root + 'ade20k_instance_train.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
