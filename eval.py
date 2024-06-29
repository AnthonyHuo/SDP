"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
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
TASK_ID= 0
taskid2cfg = {
    0 : 'config/tasks/coffee_preparation_d0.yaml',
}


def process_task(task_id,cfg):
    # delete the task0 key from the config
    new_cfg = copy.deepcopy(cfg)
    with open_dict(new_cfg):
        del new_cfg['task0']
        # add the task_id key to the config
        yaml_path = taskid2cfg[task_id]
        new_dict = {}
        with open(yaml_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        new_dict['task']=data
        new_cfg.update(new_dict)
    return new_cfg

@click.command()
@click.option('-c', '--checkpoint', default='/home/yixiao/ROBOTICS/sparse-diffusion-policy/outputs/2024-05-17/17-38-25/checkpoints/latest.ckpt')
@click.option('-o', '--output_dir', default='eval_new_tasks_output/coffee_preparation_record')
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = process_task(TASK_ID,cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    datasets=[hydra.utils.instantiate(cfg['task'].dataset)]
    normalizers=[]
    for dataset in datasets:
        normalizers.append(dataset.get_normalizer())
    workspace.model.set_normalizer(normalizers)

    policy = workspace.model
    if cfg.training.use_ema:
        workspace.ema_model.set_normalizer(normalizers)
        policy = workspace.ema_model
    policy.task_id = torch.tensor(TASK_ID, dtype=torch.int64).to(device)
    device = torch.device(device)
    policy.to(device)
    for normalizer in policy.normalizers:
        normalizer.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        max_steps=400,
        output_dir=output_dir)
    runner_log = env_runner.run(policy, task_id=torch.tensor(TASK_ID, dtype=torch.int64).to(device))
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1,'
    os.environ["MUJOCO_GL"]="osmesa"
    main()
