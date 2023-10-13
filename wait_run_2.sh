#!/bin/bash

PID=142177

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 30
bash /data2/yunfei/SpformerV1/tools/dist_train.sh /data2/yunfei/SpformerV1/configs/superformer/superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre_1group_p6_nols_conv3_loss.py 8 --cfg-options load_from=/data2/yunfei/best_pascal.pth
