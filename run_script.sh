#!/bin/bash
PARTITION="llmeval"
spot="--quotatype=spot "


export NCCL_IB_HCA="mlx5_0"
export NCCL_CROSS_NIC=0
export NCCL_IB_PCI_RELAXED_ORDERING=0
export NCCL_SOCKET_IFNAME=eth0
export DEVICE="cuda"

source /mnt/lustre/wangguoteng.p/init_env.sh

echo "$NCCL_ALGO"
echo "$NCCL_PROTO"
echo "$NCCL_MAX_NCHANNELS"
echo "$NCCL_MIN_NCHANNELS"
echo "$NCCL_NTHREADS"
echo "$USE_IB"
echo "$DATA_TYPE"
echo "$WORLD_SIZE"

touch ${JOB_NAME}
# Runs the "codeparrot-small" parameter model
if [[ "$USE_IB" == "1" ]] ; then
  export JOB_NAME_TEMP=${JOB_NAME} && \
  srun -p ${PARTITION}  ${spot} --time=12:00 -n${WORLD_SIZE} -N${WORLD_SIZE} --ntasks-per-node=1 --gpus-per-task=1  \
    /mnt/petrelfs/share_data/llm_env/miniconda3-py39_4/envs/llm-flash2.0/bin/python train.py  config/train_shakespeare_char.py  & 
else
  export JOB_NAME_TEMP=${JOB_NAME} && \
  srun -p ${PARTITION} ${spot} --time=12:00  -n${WORLD_SIZE} -N1 --ntasks-per-node=4 --gpus-per-task=1 \
    /mnt/petrelfs/share_data/llm_env/miniconda3-py39_4/envs/llm-flash2.0/bin/python train.py  config/train_shakespeare_char.py  &
fi
