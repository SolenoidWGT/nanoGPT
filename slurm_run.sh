#!/bin/bash
set -x
# Runs the "codeparrot-small" parameter model
## env config
NNODES=1

DATA_TYPE="bfloat6"
DEVICE="cuda"
WORLD_SIZE=8
GPUS_PER_NODE=8
NODE_COUNT=0
let NODE_COUNT=WORLD_SIZE/8
FOLDER_NAME=${DATA_TYPE}_${DEVICE}_${WORLD_SIZE}

if [[ ! -d ${FOLDER_NAME} ]]; then
    mkdir -p ${FOLDER_NAME}
fi

LOG_NAME="nano_${DATA_TYPE}_${DEVICE}"
LAUNCH_TIME=$(date +'%m-%d-%H:%M:%S')
log_file=${LOG_NAME}_${LAUNCH_TIME}.log
export JOB_NAME_TEMP=${LOG_NAME}-${LAUNCH_TIME}
export JOB_NAME_TEMP=${JOB_NAME_TEMP} && \
 srun -p llm_s --time=12:00  -n${WORLD_SIZE} -N${NODE_COUNT} --ntasks-per-node=${GPUS_PER_NODE} --gpus-per-task=1 \
  python train.py  config/train_shakespeare_char.py > ./${FOLDER_NAME}/${log_file} 2>&1 & 

# --quotatype=spot