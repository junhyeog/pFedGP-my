import random
import subprocess
import time


def run(cmds, n, ilow=False, type=0, sleep=5, job_name=""):
    job_name = f"{job_name}_n:{n}_l:{ilow}"
    for i in range((n - len(cmds) % n) % n):
        cmds.append(cmds[i])
    print(f"len(cmds): {len(cmds)} -> {len(cmds)/n} jobs")
    for cmd in cmds:
        print(cmd)
    for i in range(0, len(cmds), n):
        if type == 0:
            cmd = [f"sbatch -J {job_name} -n {n} -c 12 --exclude klimt,goya,haring,manet,hongdo {'--qos=ilow' if ilow else ''} --gres=gpu:normal:1 run_sbatch"]
        elif type == 1:
            cmd = [f"sbatch -J {job_name} -n {n} -c 12 {'--qos=ilow' if ilow else ''} --gres=gpu:large:1 run_sbatch"]
        else:
            # cmd = [f"sbatch -J {job_name} -n {n} -c 80  --exclude haring,vermeer {'--qos=ilow' if ilow else ''} --gres=gpu:large:1 --partition a6000 run_sbatch"]
            cmd = [f"sbatch -J {job_name} -n {n} -c 24 {'--qos=ilow' if ilow else ''} --gres=gpu:large:1 --partition a6000 run_sbatch"]

        for j in range(n):
            cmd.append("'" + cmds[i + j] + "'")
        subprocess.call(" ".join(cmd), shell=True)
        time.sleep(sleep if i + n < len(cmds) else 0)


# function that make a list of commands
# if param is list, make a list of commands for each element of the list
def make_cmds(run_file_name, params):
    cmds = []
    if isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, list):
                for v in value:
                    params[key] = v
                    cmds += make_cmds(run_file_name, params.copy())
                return cmds
        cmds.append(f"python {run_file_name}")
        for key, value in params.items():
            cmds.append(f"--{key} {value}")
        return [" ".join(cmds)]
    else:
        raise NotImplementedError


# params = {
#     "env": "bmfl",
#     "seed": 777,
#     "num-steps": 1000,
#     "num-clients": 130,
#     "num-novel-clients": 30,
#     "num-client-agg": 10,
#     # "get-data-type": [3, 4, 5, 6], # # 3: copy all test data
#     # "get-data-type": [4, 5, 6], # # 3: copy all test data
#     "get-data-type": [6], # # 3: copy all test data
#     "alpha": [0.1, 0.5, 5.0],
#     "data-name": ["cifar10", "cifar100"],
#     "test_dist": ["dirichlet"],
#     # "exp-name": "copy_all_1",
#     "exp-name": "gdt_test_1",
#     "n_trials": 16,
# }

# cmds = make_cmds("main.py", params)
# for i in range(1):
#     run(cmds, n=3, ilow=0, type=0, sleep=5, job_name=f"{params['exp-name']}_nt:{params['n_trials']}")  # type = normal: 0, large: 1, a6000: 2


# params = {
#     "env": "bmfl",
#     "seed": 777,
#     "num-steps": 1000,
#     "num-clients": 130,
#     "num-novel-clients": 30,
#     "num-client-agg": 10,
#     "data-name": ["cifar100"],
#     "alpha": [0.5],
#     "exp-name": "copy_adap_1",
#     "n_trials": 16,
# }

# cmds = make_cmds("main_adap.py", params)
# for i in range(1):
#     run(cmds, n=1, ilow=0, type=1, sleep=5, job_name=f"{params['exp-name']}_nt:{params['n_trials']}")  # type = normal: 0, large: 1, a6000: 2

# params = {
#     "env": "bmfl",
#     "seed": 777,
#     "num-steps": 1000,
#     "num-clients": 130,
#     "num-novel-clients": 30,
#     "num-client-agg": 10,
#     "alpha": [0.1, 0.5, 5.0],
#     "data-name": ["cifar100", "cifar10"],
#     "get-data-type": [0], # # 3: copy all test data
#     "test_dist": ["consistent"],
#     "exp-name": "consistent_1",
#     "n_trials": 7,
# }

# cmds = make_cmds("main.py", params)
# for i in range(1):
#     run(cmds, n=2, ilow=0, type=0, sleep=5, job_name=f"{params['exp-name']}_nt:{params['n_trials']}")  # type = normal: 0, large: 1, a6000: 2


# params = {
#     "env": "bmfl",
#     "seed": [777],
#     "num-steps": [1000],
#     "num-clients": 130,
#     "num-novel-clients": 30,
#     "num-client-agg": 10,
#     # "alpha": [5.0, 0.5, 0.1],
#     "alpha": [5.0, 5.0, 5.0],
#     "data-name": ["cifar100"],
#     "get-data-type": [10],
#     "test_dist": ["dirichlet"],
#     "exp-name": "pool_size_test_1",
#     "n_trials": 1,
# }

# cmds = make_cmds("main.py", params)
# for i in range(1):
#     run(cmds, n=3, ilow=0, type=0, sleep=5, job_name=f"{params['exp-name']}_nt:{params['n_trials']}")  # type = normal: 0, large: 1, a6000: 2


params = {
    "env": "bmfl",
    "seed": [777],
    "num-steps": [1000],
    "eval-every": [100],
    "num-clients": 130,
    "num-novel-clients": 30,
    "num-client-agg": 10,
    "data-name": ["cifar100"], # cifar10 -> normal n=1
    "get-data-type": [10],
    "test_dist": ["dirichlet"],
    "batch-size": [512], # 320
    "pool_size": [30], # 50
    "pool_type": ["full", "min"],
    "alpha": [5.0, 0.5, 0.1],
    "exp-name": "pool_is1",
    "n_trials": 1,
}

cmds = make_cmds("main.py", params)
for i in range(1):
    run(cmds, n=1, ilow=0, type=1, sleep=5, job_name=f"{params['exp-name']}_nt:{params['n_trials']}")  # type = normal: 0, large: 1, a6000: 2
