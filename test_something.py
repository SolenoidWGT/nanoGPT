import copy
import os
import re
import subprocess
import time
from collections import OrderedDict
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, List

exp_folder = "test_nccl_channel_16"
test_script = "run_script.sh"
metrics = []
valuess = []
exps_info = []

def DFS(idx, exp_info: Dict):
    if idx >= len(metrics):
        # print(exp_info, flush=True)
        exps_info.append(copy.deepcopy(exp_info))
        return

    for v in valuess[idx]:
        exp_info.update({metrics[idx]: v.strip()})
        DFS(idx+1, exp_info)

def build_log_name(info):
    name = f"./{exp_folder}/"
    for k, v in info.items():
        name += f"{v}_"
    name = name.strip("_")
    name += '.log'
    return name

def launch_exp(exp: Dict):
    timeout = 60*60
    cmd = f"bash {test_script} > {exp['JOB_NAME']} 2>&1"
    Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, env=exp)
    time.sleep(30)


def check_skip_file(file_path, target_string):
    try:
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                # 检查每一行是否包含目标字符串
                if target_string in line:
                    return True
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # 如果未找到目标字符串或发生异常，则返回False
    return False


if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

with open('test_metric.txt', 'r', encoding="utf-8") as f:
    line = f.readline().strip()
    while len(line) > 0:   
        metric, values = line.split(":")
        metrics.append(metric)
        valuess.append(list(values.split(',')))
        line = f.readline().strip()

# build exp
DFS(0, OrderedDict())

# launch exp
for exp in exps_info:
    exp_name = build_log_name(exp)
    exp.update({"JOB_NAME": exp_name})
    if check_skip_file(exp['JOB_NAME'], "iter 250:"):
        print(f"Skip exp: {exp['JOB_NAME']}", flush=True)
        continue
    print(exp)
    launch_exp(exp)

# get exp info
results = []
for file in os.listdir(exp_folder):
    fn_path = os.path.join(exp_folder, file)
    with open(fn_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "iter 250:" in line:
                loss = float(line.split(':')[1].split(',')[0].split(' ')[2])
                results.append((file, loss))
                break
    print(f"{file} loss: {loss}", flush=True)

re_set = dict()
for re in results:
    if re[1] not in re_set:
        re_set[re[1]] = [re[0]]
    else:
        re_set[re[1]].append(re[0])

for k in exps_info[0].keys():
    print(f'{k}, ', end='', flush=True)

for k,v in re_set.items():
    print(f"{k}, {v}", flush=True)


# NCCL_ALGO, NCCL_PROTO, USE_IB,
# 2.4092, ['Tree_LL_0.log', 
#          'Tree_Simple_0.log', 
#          'Ring_Simple_0.log', 
#          'Tree_LL128_0.log', 
#          'Ring_LL128_0.log', 
#          'Ring_LL_0.log', 
#          'Ring_LL_1.log', 
#          'Ring_Simple_1.log', 
#          'Tree_LL_1.log', 
#          'Tree_Simple_1.log']
# 2.394, ['Tree_LL128_1.log']
# 2.3808, ['Ring_LL128_1.log']


# 2.4092, ['Ring_LL128_16_16_0_512_0_4_float32_0.log', 'Tree_Simple_16_16_0_512_0_4_float32_0.log', 'Ring_Simple_16_16_0_512_1_4_float32_0.log', 'Ring_LL_16_16_0_512_0_4_float32_0.log', 'Tree_LL_16_16_0_512_1_4_float32_0.log', 'Tree_LL_16_16_0_512_0_4_float32_0.log', 'Ring_Simple_16_16_0_512_0_4_float32_0.log', 'Tree_LL128_16_16_0_512_0_4_float32_0.log']
# 2.3905, ['Tree_LL128_16_16_0_512_1_4_float32_0.log']
# 2.4039, ['Ring_LL128_16_16_0_512_1_4_float32_0.log']
