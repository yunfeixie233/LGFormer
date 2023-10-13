#!/bin/bash

# 定义一个数组，包含所有的配置参数
PID=691409

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done

sleep 1
echo "Process $PID has terminated"

configs=(
"/data2/yunfei/SpformerV1/configs/superformer/superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre_1group_p8_nols_avg3_loss_6h_nob.py"
"/data2/yunfei/SpformerV1/configs/superformer/superformer_baseline_voc12aug-480x480_adam_cross_2stage_sim_pred_resize_final_added_small_pre_1group_p8_nols_avgconv3_loss_6h_nob.py"
# ... 你可以在这里添加更多的配置参数 ...
)

# 遍历数组中的每一个配置参数
for config in "${configs[@]}"; do

    # 运行命令并将其放在后台运行
    bash /data2/yunfei/SpformerV1/tools/dist_train.sh "$config" 8 --cfg-options load_from=/data2/yunfei/best_pascal.pth &

    # 获取命令的PID
    PID=$!

    # 检查这个PID对应的进程是否仍在运行
    while ps -p $PID > /dev/null; do
        echo "Process $PID is still running"
        sleep 10

    done

    echo "Process $PID has terminated"

    sleep 30
done
