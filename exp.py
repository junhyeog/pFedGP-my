import random
import subprocess
import time


def run(cmds, n, ilow=False, type=0, sleep=5):
    for i in range((n - len(cmds) % n) % n):
        cmds.append(cmds[i])
    print(f"len(cmds): {len(cmds)} -> {len(cmds)/n} jobs")
    for cmd in cmds:
        print(cmd)
    for i in range(0, len(cmds), n):
        if type == 0:
            cmd = [f"sbatch -n {n} -c 8 --exclude klimt,goya,haring,manet,hongdo {'--qos=ilow' if ilow else ''} --gres=gpu:normal:1 run_sbatch"]
        elif type == 1:
            cmd = [f"sbatch -n {n} -c 12 --exclude cerny,kandinsky,namjune,magritte {'--qos=ilow' if ilow else ''} --gres=gpu:large:1 run_sbatch"]
        else:
            # cmd = [f"sbatch -n {n} -c 80  --exclude haring,vermeer {'--qos=ilow' if ilow else ''} --gres=gpu:large:1 --partition a6000 run_sbatch"]
            cmd = [f"sbatch -n {n} -c 24 {'--qos=ilow' if ilow else ''} --gres=gpu:large:1 --partition a6000 run_sbatch"]

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

# optimizer, wd

# cifar10, bs = 64
params = {
    ### optinal
    # "optimizer": ["sgd", "adam"],
    "wd": [0.001, 0.0005, 0.0],
    ### optinal
    "data-path": "../../datafolder",
    "save-path": "../output/pFedGP",
    "seed": "777",
    "num-clients": "130",
    "num-novel-clients": "30",
    "num-client-agg": "10",
    "data-name": ["cifar10"],
    "alpha": [0.1],
    "inner-steps" :[5], # 1 / 5
    "lr": [0.03, 0.025, 0.02, 0.015], # 5e-2 / 0.03
    "batch-size": [64], # 512 / 64
    "exp-name": ["bmfl1"],
}
cmds = make_cmds("experiments/ood_generalization/trainer.py", params)
run(cmds, n=3, ilow=0, type=0, sleep=5)

# cifar100, bs = 64
params = {
    ### optinal
    # "optimizer": ["sgd", "adam"],
    # "wd": [0.001, 0.0005, 0.0],
    ### optinal
    "data-path": "../../datafolder",
    "save-path": "../output/pFedGP",
    "seed": "777",
    "num-clients": "130",
    "num-novel-clients": "30",
    "num-client-agg": "10",
    "data-name": ["cifar100"],
    "alpha": [0.5, 5.0],
    "inner-steps" :[5], # 1 / 5
    "lr": [0.03, 0.025, 0.02, 0.015], # 5e-2 / 0.03
    "batch-size": [64], # 512 / 64
    "exp-name": ["bmfl1"],
}
cmds = make_cmds("experiments/ood_generalization/trainer.py", params)
run(cmds, n=1, ilow=0, type=0, sleep=5)
