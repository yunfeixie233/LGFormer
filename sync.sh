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

default_load_from="small_conv_ade.pth"

# Check for the --resume argument
resume=false
resume_from=""
load_from=""

if [ "$1" == "--resume" ]; then
    resume=true
    resume_from=$2
    load_from="${3:-$default_load_from}"
else
    load_from="${1:-$default_load_from}"
fi

# Sync interval in seconds (e.g., 3600 seconds = 1 hour)
SYNC_INTERVAL=300

# Define remote host information
REMOTE_USER="meijieru"
REMOTE_HOST="8.210.160.55"
REMOTE_BASE_DIR="/data1/yunfei/log"
REMOTE_PORT=20046

configs=(
    "/data2/yunfei/SpformerV1/configs/superformer/city/gt/pixel_extra_small_pre_p7p8p9p10_nols_lea16_fusesp_fusegt_noscale.py"
)
should_skip_sync() {
    local skip_sync=false
    # 获取本机的所有 IP 地址
    local ips=$(hostname -I)
    for ip in $ips; do
        if [[ "$ip" == "$REMOTE_HOST" ]]; then
            skip_sync=true
            break
        fi
    done
    echo $skip_sync
}

get_server_name() {
    local hostname_str=$(hostname)
    local server_name=""

    if [[ "$hostname_str" == *autodl* ]]; then
        # 根据hostname中包含的autodl特定内容来设置server_name
        case "$hostname_str" in
            *a6b311913c-660df582*)
                server_name="autodl-k"
                ;;
            *efb1119a8e-b265940c*)
                server_name="autodl-i"
                ;;
            # 在这里添加更多的autodl相关的case匹配规则
            *)
                server_name="$hostname_str"  # 如果没有特定匹配，使用hostname本身
                ;;
        esac
    else
        # 如果hostname中不包含autodl，直接使用hostname
        server_name="$hostname_str"
    fi

    echo "$server_name"
}

sync_and_cleanup() {
    local SRC_DIR="$1"
    local DATE_PREFIX=$(date +%m%d)

    # 我们需要找到 "work_dirs" 字符串的位置，然后提取它之前的所有内容
    local LOCAL_BASE_DIR=$(echo "$SRC_DIR" | sed 's|\(.*\)/work_dirs/.*|\1|')
    # 提取 "work_dirs" 之后的路径（不包括 "work_dirs" 本身），这将成为我们的相对路径
    local RELATIVE_PATH="${SRC_DIR#*$LOCAL_BASE_DIR/work_dirs/}"

    # 去掉配置文件的扩展名 '.py'
    RELATIVE_PATH="${RELATIVE_PATH%.py}"

    # 拼接远程目录路径，去除开始的 'configs/'
    local REMOTE_DIR="${REMOTE_BASE_DIR}/${RELATIVE_PATH#configs/}"

    local DEST_DIR_NAME="${DATE_PREFIX}_$(basename "$SRC_DIR" .py)"
    local TMP_DIR="/tmp/${DEST_DIR_NAME}"
    mkdir -p "$TMP_DIR"
    rsync -av --exclude='*.pth' "$SRC_DIR/" "$TMP_DIR" > /dev/null 2>&1
    
    # rsync -av --delete -e "ssh -p ${REMOTE_PORT}" "$TMP_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" > /dev/null 2>&1
    rsync -av --delete --inplace --rsync-path="mkdir -p ${REMOTE_DIR} && rsync" -e "ssh -p ${REMOTE_PORT}" "$TMP_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

    rm -rf "$TMP_DIR"
    echo "Syncing $WORK_DIR to $REMOTE_DIR "    


}




first=true
while true; do  
    for config in "${configs[@]}"; do
        server_name=$(get_server_name)  
        WORK_DIR_BASENAME=$(basename "${config%.*}")
        WORK_DIR="${config/configs/work_dirs}"
        WORK_DIR="${WORK_DIR%.py}"        

        if $first && $resume; then
            echo "Resuming training from $resume_from for $config"
            bash tools/dist_train.sh "$config" 8 --work-dir "$WORK_DIR" --resume --cfg-options load_from="$resume_from" randomness.diff_rank_seed=False randomness.seed=1539460459 &
            first=false
        else
            echo "Starting training from $load_from for $config"
            bash tools/dist_train.sh "$config" 8 --work-dir "$WORK_DIR" --cfg-options load_from="$load_from" randomness.diff_rank_seed=False randomness.seed=1539460459 &
        fi

        python reminder.py "$config" 1 "$server_name"  
        PID=$!
        LAST_SYNC_TIME=$(date +%s)
        while ps -p $PID > /dev/null; do
            echo "Process $PID is still running"
            CURRENT_TIME=$(date +%s)
            if (( CURRENT_TIME - LAST_SYNC_TIME >= SYNC_INTERVAL )); then
                sync_and_cleanup "$WORK_DIR"
                LAST_SYNC_TIME=$CURRENT_TIME
            fi
            sleep 10
        done

        echo "Process $PID has terminated"


        python reminder.py "$config" 0 "$server_name"  
        sync_and_cleanup "$WORK_DIR"
        sleep 30
    done
    python reminder.py "所有" 0 "$server_name"  
    echo "All configurations processed."
done