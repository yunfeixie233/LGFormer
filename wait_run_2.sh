#!/bin/bash

PID=67775

while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 1
bash /data2/yunfei/SpformerV1/tools/dist_train.sh/data2/yunfei/SpformerV1/configs/superformer/ade/nogt/sp_extra_small_pre.py 8 
