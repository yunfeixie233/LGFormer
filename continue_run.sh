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



PID=2539867

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done

sleep 10
echo "Process $PID has terminated"

configs=(
    "/data2/yunfei/SpformerV1/configs/superformer/ade/nogt/sp_regproxy_small_pre_6sim.py"
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
