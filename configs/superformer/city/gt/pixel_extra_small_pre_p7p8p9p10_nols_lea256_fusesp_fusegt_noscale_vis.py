_base_ = [
    './pixel_extra_small_pre_p7p8p9p10_nols_lea256_fusesp_fusegt.py',
]
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    group_pos = ((),(5,6,7,8,),()),        
    group_qk_scale = 1,
    vis_gt_eff = True,
    vis_sp = True,
    vis_gt = True,  
    output_dir = '/data2/yunfei/pixel_extra_small_pre_p7p8p9p10_nols_lea256_fusesp_fusegt_noscale_vis'      
    
))
