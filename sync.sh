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

# Check for the --resume argument
resume=false
resume_from=""
if [ "$1" == "--resume" ]; then
    resume=true
    resume_from=$2
fi

load_from="${3:-best_coco.pth}"

# Sync interval in seconds (e.g., 3600 seconds = 1 hour)
SYNC_INTERVAL=300

# Define remote host information
REMOTE_USER="meijieru"
REMOTE_HOST="169.233.1.28"
REMOTE_BASE_DIR="/data2/yunfei/SpformerV1/work_dirs"

configs=(
    "/data2/yunfei/SpformerV1/configs/superformer/pascal/gt/sp_extra_small_pre_p9_ls_avg4_fuseconv.py"
    "/data2/yunfei/SpformerV1/configs/superformer/pascal/gt/pixel_extra_small_pre_p9_ls_avg4.py"
    "/data2/yunfei/SpformerV1/configs/superformer/pascal/gt/pixel_extra_small_pre_p9_ls_avg4_fusesp.py"
    "/data2/yunfei/SpformerV1/configs/superformer/pascal/gt/pixel_extra_small_pre_p9_ls_avg4_fusesp_fusegt.py"    

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
    local server_ip=$(hostname -I | awk '{for(i=1;i<=NF;i++) if ($i ~ /^169\./) print $i}')
    local server_name=""

    case "$server_ip" in
        "169.233.1.28")
            server_name="4u-7"
            ;;
        "169.233.1.27")
            server_name="4u-6"
            ;;
        "169.233.1.18")
            server_name="4u-1"
            ;;            
        # 可以在此添加更多的匹配规则
        *)
            server_name="UnknownServer"
            ;;
    esac

    echo "$server_name"
}

sync_and_cleanup() {
    local SRC_DIR="$1"
    local DATE_PREFIX=$(date +%m%d)
    local DEST_DIR_NAME="${DATE_PREFIX}_$(basename "$SRC_DIR")"
    local TMP_DIR="/tmp/${DEST_DIR_NAME}"

    mkdir -p "$TMP_DIR"
    rsync -av --exclude='*.pth' "$SRC_DIR/" "$TMP_DIR" > /dev/null 2>&1
    rsync -av --delete "$TMP_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/${DEST_DIR_NAME}" > /dev/null 2>&1
    rm -rf "$TMP_DIR"
}

first=true
for config in "${configs[@]}"; do
    server_name=$(get_server_name)  
    WORK_DIR_BASENAME=$(basename "${config%.*}")
    WORK_DIR="/data2/yunfei/SpformerV1/work_dirs/${WORK_DIR_BASENAME}"

    if $first && $resume; then
        echo "Resuming training from $resume_from for $config"
        bash tools/dist_train.sh "$config" 8 --resume --cfg-options load_from="$resume_from" randomness.diff_rank_seed=False randomness.seed=1539460459 &
        first=false
    else
        echo "Starting training from $load_from for $config"
        bash tools/dist_train.sh "$config" 8 --cfg-options load_from="$load_from" randomness.diff_rank_seed=False randomness.seed=1539460459 &
    fi

    python /data2/yunfei/SpformerV1/reminder.py "$config" 1 "$server_name"  
    PID=$!

    LAST_SYNC_TIME=$(date +%s)
    while ps -p $PID > /dev/null; do
        echo "Process $PID is still running"
        CURRENT_TIME=$(date +%s)
        if (( CURRENT_TIME - LAST_SYNC_TIME >= SYNC_INTERVAL )); then
            echo "Syncing $WORK_DIR"
            sync_and_cleanup "$WORK_DIR"
            LAST_SYNC_TIME=$CURRENT_TIME
        fi
        sleep 10
    done

    echo "Process $PID has terminated"


    python /data2/yunfei/SpformerV1/reminder.py "$config" 0 "$server_name"  
    sync_and_cleanup "$WORK_DIR"
    sleep 30
done
python /data2/yunfei/SpformerV1/reminder.py "所有" 0 "$server_name"  
echo "All configurations processed."
