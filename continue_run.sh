#!/bin/bash
unset LD_LIBRARY_PATH

desired_env="yunfei_v2"
current_env=$(conda env list | grep '*' | awk '{print $1}')

if [[ "$current_env" != "$desired_env" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate $desired_env
    current_env=$(conda env list | grep '*' | awk '{print $1}')
    echo "Switched to env $current_env"
fi

# 设置默认的 load_from 值，如果提供了参数则使用第一个参数
load_from="${1:-best_ade.pth}"

PID=3369913

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done

sleep 10
echo "Process $PID has terminated"

configs=(
    "/data2/yunfei/SpformerV1/configs/superformer/ade/gt/sp_extra_small_pre_p6_ls_avg4_loss_6h_expandgt_groupcls.py"
    
)

while true; do  
    for config in "${configs[@]}"; do

        bash tools/dist_train.sh \
        "$config" 8 --cfg-options load_from="$load_from" \
        randomness.diff_rank_seed=False \
        randomness.seed=1539460459 &
        echo "load_from $load_from"
        PID=$!

        while ps -p $PID > /dev/null; do
            echo "Process $PID is still running"
            sleep 10
        done

        echo "Process $PID has terminated"
        sleep 30
    done

    last_config=${configs[-1]}
    bash tools/dist_train.sh \
    "$config" 8 --cfg-options load_from="$load_from" \
    randomness.diff_rank_seed=False \
    randomness.seed=1539460459 &

    PID=$!

    while ps -p $PID > /dev/null; do
        echo "Process $PID is still running"
        sleep 10
    done

    echo "Process $PID has terminated"
    sleep 30
done
