#!/bin/bash

folder_path="/data2/yunfei/SpformerV1/work_dirs/"

# 遍历每个子文件夹
find "$folder_path" -type d | while read -r dir; do
    # 找出需要保留的两个文件
    keep_files=$(find "$dir" -maxdepth 1 -type f -name "*.pth" | sed 's/.*iter_\([0-9]*\)\.pth/\1 &/' | sort -nr | awk '{print $2}' | head -n 2)

    # 打印出这些文件
    echo "Keeping files in $dir:"
    echo "$keep_files"

    # 删除除了这两个文件以外的所有.pth文件
    find "$dir" -maxdepth 1 -type f -name "*.pth" | grep -vFf <(echo "$keep_files") | while read -r file; do
        rm "$file"
    done
done
