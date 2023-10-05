#!/bin/bash

# 如果不提供足够的参数，显示用法信息
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <source_directory> <time_interval_in_seconds>"
    exit 1
fi

# 从命令行参数获取源文件夹路径和时间间隔
SOURCE_DIR=$1
INTERVAL=$2

while true; do
    TMP_DIR="/tmp/my_folder_$(date +%s)"
    rsync -av --exclude='*.pth' $SOURCE_DIR $TMP_DIR
    scp -rP 22  $TMP_DIR  meijieru@169.233.1.30:/data2/yunfei/sync/SpformerV1/work_dirs

    # 暂停指定的时间间隔
    sleep $INTERVAL
done

