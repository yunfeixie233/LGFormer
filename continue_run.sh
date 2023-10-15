#!/bin/bash
unset LD_LIBRARY_PATH

PID=1951438

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done

sleep 1
echo "Process $PID has terminated"

configs=(
    "/data2/yunfei/SpformerV1/configs/superformer/ade/gt/sp_extra_small_pre_p6_ls_avg4_loss_6h_expandgt.py"
    "/data2/yunfei/SpformerV1/configs/superformer/ade/gt/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_nob.py"
)

while true; do  
    for config in "${configs[@]}"; do

        bash /data2/yunfei/SpformerV1/tools/dist_train.sh \
        "$config" 8 --cfg-options load_from=/data2/yunfei/best_ade.pth \
        randomness.diff_rank_seed=False \
        randomness.seed=3407 &

        PID=$!

        while ps -p $PID > /dev/null; do
            echo "Process $PID is still running"
            sleep 10

        done

        echo "Process $PID has terminated"
        sleep 30

    done

    last_config=${configs[-1]}
    bash /data2/yunfei/SpformerV1/tools/dist_train.sh \
    "$config" 8 --cfg-options load_from=/data2/yunfei/best_ade.pth \
    randomness.diff_rank_seed=False \
    randomness.seed=3407 &

    PID=$!

    while ps -p $PID > /dev/null; do
        echo "Process $PID is still running"
        sleep 10
    done

    echo "Process $PID has terminated"
    sleep 30

done  
