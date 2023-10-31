#!/bin/bash

unset LD_LIBRARY_PATH

PID=2785534
while ps -p $PID > /dev/null; do
    echo "Process $PID is still running"
    sleep 10
done


echo "Process $PID has terminated"


sleep 30
python -m torch.distributed.launch --nproc_per_node 8 train.py --launcher pytorch --config configs/regproxy_ade20k/regproxy-s16-sub11+implicit-mid-4+512x512+80k+adamw-poly+ade20k.py  --seed 42 --deterministic    --options model.pretrained='/data2/yunfei/vit_small_patch16_384_pretrain.pth'  
