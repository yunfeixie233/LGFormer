_base_ = [
    '../158/gt_cls/gt_extra_small_pre_p9p10p11_nols_avg4_all.py',
    '../../../_base_/datasets/partimagenet_part.py',    
]
num_classes = 57
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
    seg_num_classes = 41,     
))
