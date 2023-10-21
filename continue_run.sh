#!/bin/bash
unset LD_LIBRARY_PATH

desired_env="yunfei_v2"


current_env=$(conda env list | grep '*' | awk '{print $1}')

if [[ "$current_env" != "$desired_env" ]]; then

    eval "$(conda shell.bash hook)"
    conda activate $desired_env
    current_env=$(conda env list | grep '*' | awk '{print $1}')
    echo switch to env $current_env
fi



PID=595532

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done

sleep 1
echo "Process $PID has terminated"

configs=(
    "/root/autodl-tmp/SpformerV1/configs/superformer/ade/gt/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_multi_concat.py"
)

while true; do  
    for config in "${configs[@]}"; do

        bash tools/dist_train.sh \
        "$config" 8 --cfg-options load_from=best_ade.pth \
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

    last_config=${configs[-1]}
    bash tools/dist_train.sh \
    "$config" 8 --cfg-options load_from=best_ade.pth \
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
