#!/bin/bash

PID=110744

while kill -0 $PID 2> /dev/null; do
    sleep 5  # 每5秒检查一次
done

bash /root/autodl-tmp/study/SpformerV1/tools/dist_train.sh /root/autodl-tmp/study/SpformerV1/configs/superformer/superformer_baseline_ade20k-512x512_adam_cross_2stage_sim_pred_resize_final_added_pre_v3_2group_conv_1iter.py 8 --cfg-options load_from=/root/autodl-tmp/v2.pth
