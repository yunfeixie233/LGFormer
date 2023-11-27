_base_ = [
    './gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
]
num_classes = 57
model = dict(
    
    decode_head=dict(
    group_init_strides = (2,2,2,),
    group_init_kernel_sizes = (2,2,2,),
    group_layers = {0:256,1:256,2:256}, 
    output_dir = "/data1/yunfei/gt_extra_small_pre_p9p10p11_nols_avg2_1loss_test",
    vis_sp_id = True,             
    use_final_group_cls = True,
    ),
 test_cfg=dict(mode='whole'))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]   
# test_dataloader =val_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline))    

test_dataloader =val_dataloader = dict(
    dataset=dict(

        pipeline=test_pipeline))