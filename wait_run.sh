#!/bin/bash

unset LD_LIBRARY_PATH

PID=1129196
while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 30
bash tools/dist_train.sh /data2/yunfei/study/SpformerV1/configs/superformer/ade/nogt/pixel_extra_small_pre.py 8
