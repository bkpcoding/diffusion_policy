import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_image_runner import AdversarialRobomimicImageRunner
from omegaconf import OmegaConf
# torch.backends.cudnn.enabled = False
@hydra.main(config_path='diffusion_policy/eval_configs', config_name='ibc_image_ph_pick_adversarial')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_adversarial')
def main(cfg):
    checkpoint = cfg.checkpoint
    task = cfg.task
    algo = cfg.algo
    n_envs = cfg.n_envs
    device = cfg.device
    attack = cfg.attack
    log = cfg.log
    epsilons = cfg.epsilons
    dataset_path = cfg.dataset_path
    view = cfg.view

    if log:
        wandb.init(project='BC_Evaluation', name=f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-\
        {checkpoint.split("/")[-3]}-adversarial_on_{view}' if attack else\
        f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-{checkpoint.split("/")[-3]}')

    for epsilon in epsilons:
        # the output directory should depend on the current directory and the checkpoint path and the attack type and epsilon
        output_dir = os.path.join(os.getcwd(), f"diffusion_policy/data/experiments/image/{task}/{algo}/eval_{checkpoint.split('/')[-3]}_{epsilon}_{view}")
        if os.path.exists(output_dir):
            raise ValueError(f"Output path {output_dir} already exists!")

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg_loaded = payload['cfg']
        cls = hydra.utils.get_class(cfg_loaded._target_)
        workspace = cls(cfg_loaded, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        print(policy)

        if attack:
            print("Running adversarial Attack")
            cfg_loaded.task.env_runner['_target_'] = cfg._target_
            cfg_loaded.task.env_runner['n_envs'] = n_envs
        cfg_loaded.task.env_runner['dataset_path'] = str(dataset_path)

        try:
            if cfg_loaded.training.use_ema:
                policy = workspace.ema_model
        except:
            pass

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        env_runner = hydra.utils.instantiate(
            cfg_loaded.task.env_runner,
            output_dir=output_dir)
        if attack:
            runner_log = env_runner.run(policy, epsilon, cfg)
        else:
            runner_log = env_runner.run(policy)

        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        if log:
            wandb.log({"test/mean_score": json_log["test/mean_score"], "train/mean_score": json_log["train/mean_score"], \
                "Epsilon":float(epsilon)})
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    wandb.finish()

if __name__ == '__main__':
    main()