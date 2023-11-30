_base_ = [
    '../nogt/sp_extra_small_pre.py',
]
model = dict(
    type='EncoderDecoderJoint',
    decode_head=dict(
    use_gt_fuse = True,
    use_sp_fuse = True,  
    group_pos = ((),(7,),()), 
    use_group_token = 'mix',
    use_gt_loss = True,
    group_init_strides = (4,),
    group_init_kernel_sizes = (4,),
    group_layers = {0:64,},
    loss_decode=[
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_obj',use_sigmoid=False, loss_weight=0.5),
            dict(
            type='CrossEntropyLoss',loss_name = 'loss_part', use_sigmoid=False, loss_weight=0.5)],
    output_dir = "/data1/yunfei/pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt_vis",
    vis_gt = True,
    vis_sp = True,
    vis_spgt = True,
    vis_sp_id = True,               
))
