"""
Usage:
python eval.py --checkpoint /path/to/ckpt -o /path/to/output_dir
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import copy
from omegaconf.omegaconf import open_dict
import yaml

taskid2cfg = {
    0 :"config/tasks/square_d0.yaml" ,
    1 :"config/tasks/stack_d0.yaml" ,
    2 :"config/tasks/coffee_d0.yaml" ,
    3 :"config/tasks/hammer_cleanup_d0.yaml" ,
    4 :"config/tasks/mug_cleanup_d0.yaml" ,
    5 :"config/tasks/nut_assembly_d0.yaml" ,
    6 :"config/tasks/stack_three_d0.yaml" ,
    7: "config/tasks/threading_d0.yaml" ,
}



@click.command()
@click.option('-c', '--checkpoint', default='epoch=0299-test_mean_score=6.070.ckpt')
@click.option('-o', '--output_dir', default='test_eval')
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    for i in range(cfg['task_num']):
        curr_cfg=taskid2cfg[i]
        with open(curr_cfg, "r") as f:
            task_cfg = yaml.safe_load(f)
            cfg[f"task{i}"]=task_cfg
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # run eval
    # configure env
    env_runners = []
    # env_runner3: BaseImageRunner
    for i in range(cfg.task_num):
        env_runners.append(hydra.utils.instantiate(cfg[f'task{i}'].env_runner, output_dir=output_dir))


    # get policy from workspace
    datasets= []
    for i in range(cfg.task_num):
        datasets.append(hydra.utils.instantiate(cfg[f'task{i}'].dataset))
    normalizers=[]
    for dataset in datasets:
        normalizers.append(dataset.get_normalizer())
    workspace.model.set_normalizer(normalizers)

    policy = workspace.model
    if cfg.training.use_ema:
        workspace.ema_model.set_normalizer(normalizers)
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    for normalizer in policy.normalizers:
        normalizer.to(device)
    policy.eval()
    
    
    runner_logs = []
    for i, env_runner in enumerate(env_runners):
        runner_log = env_runner.run(policy,task_id=torch.tensor([i], dtype=torch.int64).to(device))
        runner_log = {key + f'_{i}': value for key, value in runner_log.items()}
        runner_logs.append(runner_log)
    
    # dump log to json
    for i,runner_log in enumerate(runner_logs):
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        out_path = os.path.join(output_dir, f'eval_log_{i}.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1,'
    os.environ["MUJOCO_GL"]="osmesa"
    main()
