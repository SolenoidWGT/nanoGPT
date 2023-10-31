#!/bin/bash
set -x
# Runs the "codeparrot-small" parameter model
## env config
GPUS_PER_NODE=8
NNODES=1

LOG_NAME="nano_float32"
LAUNCH_TIME=$(date +'%m-%d-%H:%M:%S')
log_file=${LOG_NAME}_${LAUNCH_TIME}.log
export JOB_NAME_TEMP=${LOG_NAME}-${LAUNCH_TIME}

#  
export JOB_NAME_TEMP=${JOB_NAME_TEMP} &&  srun -p llm_t --quotatype=spot -n8 -N1 --ntasks-per-node=8 --gpus-per-task=1  python train.py  config/train_shakespeare_char.py > ./nano_exp/${log_file} 2>&1 & 
