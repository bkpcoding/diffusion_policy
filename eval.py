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
from diffusion_policy.env_runner.robomimic_image_runner import AdversarialRobomimicImageRunner

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-l', '--log', default='False')
@click.option('-d', '--device', default='cuda:0')
@click.option('-a', '--attack', default='False')
@click.option('-e', '--epsilon', default=0.1)
def main(checkpoint, output_dir, device, attack, log, epsilon):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    if log == 'True':
        # the project name should depend on the checkpoint path and the attack type
        # if the checkpoint path is --checkpoint /teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/lstm_gmm/train_2/checkpoints/epoch=0300-test_mean_score=1.000.ckpt
        # the project name should be image-lift_ph-lstm_gmm-train_2-adversarial if attack is True
        # or image-lift_ph-lstm_gmm-train_2 if attack is False
        wandb.init(project='BC_Evaluation', name=f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-\
        {checkpoint.split("/")[-3]}-adversarial_{epsilon}' if attack == 'True' else\
        f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-{checkpoint.split("/")[-3]}')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    cfg.task.env_runner['n_envs'] = 1

    # if attack then change the environment to adversarial environment
    if attack == 'True':
        print("Running adversarial Attack")
        cfg.task.env_runner['_target_'] = 'diffusion_policy.env_runner.robomimic_image_runner.AdversarialRobomimicImageRunnerIBC'
        cfg.task.env_runner['n_envs'] = 1
    # Get the absolute path of the current directory
    current_dir = os.getcwd()

    # Construct the file path dynamically by joining the current directory path with the relative path
    dataset_path = os.path.join(current_dir, cfg.task.env_runner['dataset_path'])
    cfg.task.env_runner['dataset_path'] = str(dataset_path)
    try:
        if cfg.training.use_ema:
            policy = workspace.ema_model
    except:
        pass
    print(policy, cfg.task.env_runner)

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    if attack == 'True':
        runner_log = env_runner.run(policy, epsilon)
    else:
        runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    if log == 'True':
        wandb.log({"test/mean_score": json_log["test/mean_score"], "train/mean_score": json_log["train/mean_score"]})
        wandb.finish()
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
