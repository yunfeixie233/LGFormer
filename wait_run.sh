#!/bin/bash

unset LD_LIBRARY_PATH

PID=2432253
while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 30
bash tools/dist_train.sh \
/data2/yunfei/study/SpformerV1/configs/superformer/ade/gt/sp_extra_small_pre_p8_ls_lea4_loss_6h_expandgt_multi_concat.py 8 \
 --resume --cfg-options load_from=/data2/yunfei/iter_28000.pth \
randomness.diff_rank_seed=False \
randomness.seed=1539460459
