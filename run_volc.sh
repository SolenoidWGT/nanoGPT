export LOG_NAME="nano"
export MODEL_SIZE="GPT_exp"
export saved_yaml_path=Res_${MODEL_SIZE}${LOG_NAME}.yaml
export NNODES=1
export EXAMPLE_YAML_PATH="/fs-computility/llm/shared/volc_example.yaml"
export ResourceQueueName="llm"
export Tags="Test"
LAUNCH_TIME=$(date +'%m-%d-%H:%M:%S')


log_file=${MODEL_SIZE}/${LOG_NAME}_${LAUNCH_TIME}.log
let GPU_NUMS=NNODES*8
export JOB_NAME=${LOG_NAME}
export JOB_NAME_TEMP=${LOG_NAME}-${LAUNCH_TIME}

mkdir -p $MODEL_SIZE

function do_volc()
{
    set +x
    echo "do_volc (only support job whose worldsize % 8 == 0 or worldsize < 8)"
    if [[ $GPU_NUMS -lt 8 ]]; then
        num_nodes=1
        num_tasks_per_node=${GPU_NUMS}
        let node_mems=1100/GPU_NUMS
        let cpu_nums=120/GPU_NUMS
        shared_memory="10Gi"
    else
        let num_nodes=GPU_NUMS/8
        num_tasks_per_node=8
        node_mems=1100
        cpu_nums=120
        shared_memory="100Gi"
    fi

    export RoleReplicas=${num_nodes}
    export Entrypoint="export JOB_NAME="${JOB_NAME}" && \
export JOB_NAME_TEMP=${JOB_NAME_TEMP} && \
export USER=${USER} && \
export NCCL_DEBUG="" && \
/fs-computility/llm/shared/llm-flash2.0/bin/torchrun --master_addr=\$MLP_WORKER_0_HOST  \
--master_port=\$MLP_WORKER_0_PORT   \
--nproc_per_node=\$MLP_WORKER_GPU \
--nnodes=\$MLP_WORKER_NUM  \
--node_rank=\$MLP_ROLE_INDEX /fs-computility/llm/shared/wangguoteng/nanoGPT/train.py /fs-computility/llm/shared/wangguoteng/nanoGPT/config/train_shakespeare_char.py > ${log_file} 2>&1  "

    eval /fs-computility/llm/shared/llm-flash2.0/bin/python load_volc_yaml.py
    sleep 2
    cat ${saved_yaml_path}  && volc ml_task submit -c ${saved_yaml_path}
}

do_volc