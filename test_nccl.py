import torch
import random
import os
import numpy as np
import torch.distributed as dist

backend = "nccl"
device= "cuda"
DTYPE = torch.bfloat16
PREFIX = "NCCL_TEST"
COUNT_BIG = 5000
COUNT_SMALL = 50000

# set seed
seed = 1024
cuda_deterministic = True
seed_offset = 0 # 我们强制让所有的东西都一样
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.use_deterministic_algorithms(True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if cuda_deterministic:  # slower, more reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


if 'SLURM_PROCID' in os.environ:
    def get_master_node():
        import subprocess

        if os.getenv("SLURM_JOB_ID") is None:
            raise RuntimeError("get_master_node can only used in Slurm launch!")
        result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
        result = result.decode("utf8").strip()
        return result

    os.environ['MASTER_ADDR'] = f"{get_master_node()}"  #tcp://
    os.environ['MASTER_PORT'] = "12349"
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['LOCAL_RANK'] = str(int(os.environ['SLURM_PROCID']) % 8)
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NPROCS']
    # init_process_group(backend=backend)
    dist.init_process_group(
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']),
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        )

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    if device != "cpu":
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
else:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    if device != "cpu":
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

dist.barrier()
print("Init procss group done!", flush=True)

f = open(f"./{PREFIX}/RANK_{dist.get_rank()}", 'w+')

A = torch.rand(size=(8*1024, 8*1024), dtype=DTYPE).cuda().div_(4)
B = torch.rand(size=(8*1024, 8*1024), dtype=DTYPE).cuda().div_(4)
def benchmark(i, A_):
    A_ = torch.mul(A_, B)
    torch.cuda.synchronize()

    a1 = torch.sum(A_.to(torch.float64))
    torch.cuda.synchronize()

    dist.all_reduce(A_)
    torch.cuda.synchronize()

    a2 = torch.sum(A_.to(torch.float64))
    torch.cuda.synchronize()

    A_.div_((i+2)*dist.get_world_size())
    torch.cuda.synchronize()

    a3 = torch.sum(A_.to(torch.float64))
    torch.cuda.synchronize()
    if i % 100 == 0:
        print(f"a1: {a1}, a2: {a2}, a3: {a3}", file=f, flush=True)
    return A_

print(f"-----------big-----------", file=f, flush=True)

for i in range(COUNT_BIG):
    A = benchmark(i, A)


print(f"-----------small-----------", file=f, flush=True)
torch.cuda.empty_cache()

A = torch.rand(size=(8*64, 8*64), dtype=DTYPE).cuda().div_(4)
B = torch.rand(size=(8*64, 8*64), dtype=DTYPE).cuda().div_(4)

for i in range(COUNT_SMALL):
    A = benchmark(i, A)


f.close()

