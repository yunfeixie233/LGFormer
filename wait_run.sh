#!/bin/bash

PID=29045
=======
PID=629197

>>>>>>> Stashed changes
=======
PID=629197

>>>>>>> Stashed changes
while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 30
<<<<<<< Updated upstream
<<<<<<< Updated upstream
bash /root/autodl-tmp/SpformerV1/tools/dist_train.sh /root/autodl-tmp/SpformerV1/configs/superformer/superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre_2group_p68_nols_conv33_noproj.py 8 --cfg-options load_from=/root/autodl-tmp/best_pascal.pth
=======
bash /data2/yunfei/SpformerV1/tools/dist_train.sh /data2/yunfei/SpformerV1/configs/superformer/superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre_1group_p8_nols_avg3_loss_6h_nob.py 8 --cfg-options load_from=/data2/yunfei/best_pascal.pth
>>>>>>> Stashed changes
=======
bash /data2/yunfei/SpformerV1/tools/dist_train.sh /data2/yunfei/SpformerV1/configs/superformer/superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre_1group_p8_nols_avg3_loss_6h_nob.py 8 --cfg-options load_from=/data2/yunfei/best_pascal.pth
>>>>>>> Stashed changes
