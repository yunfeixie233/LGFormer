#!/bin/bash
unset LD_LIBRARY_PATH
PID=322128

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 30
bash /data2/yunfei/SpformerV1/tools/dist_train.sh /data2/yunfei/SpformerV1/configs/superformer/ade/gt/sp_extra_small_pre_p8_ls_avg4_loss_6h_expandgt_cls.py 8 --cfg-options load_from=/data2/yunfei/best_ade.pth
