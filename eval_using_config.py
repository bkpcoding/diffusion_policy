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
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate
from hydra.core.global_hydra import GlobalHydra


# torch.backends.cudnn.enabled = False
@hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_ph_pick_pgd_adversarial')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_adversarial')
def main(cfg):
    checkpoint = cfg.checkpoints[0]
    task = cfg.task
    algo = cfg.algo
    n_envs = cfg.n_envs
    device = cfg.device
    attack = cfg.attack
    log = cfg.log
    epsilons = cfg.epsilons
    if cfg.epsilon is not None:
        epsilons = [cfg.epsilon]
    dataset_path = cfg.dataset_path
    view = cfg.view

    if log:
        # wandb.init(project='BC_Evaluation', name=f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-\
        #     {checkpoint.split("/")[-3]}-{cfg.attack_type}_adversarial_on_{view}_randtar_{cfg.rand_target}' if attack else
        #     f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-{checkpoint.split("/")[-3]}')
        # wandb.init(project='ibc_pgd_experimentation', name=f'epsilon-{cfg.epsilons[0]}-rand_target-{cfg.rand_target}-rand_init-{cfg.rand_int}')
        # wandb.init(project='ibc_pgd_experimentation', name=f'epsilon-{cfg.epsilons[0]}-target_perturbations-{cfg.target_perturbations}-pertubation-{cfg.perturbations[1]}')
        # wandb.init(project="BC_Evaluation", id='skjusrmy', resume='must')
        wandb.init(project='diffusion_experimentation', name=f'diffusion_policy')
        config_path = 'diffusion_policy/eval_configs'
        config_name = 'diffusion_policy_image_ph_pick_pgd_adversarial'
        config_file_path = to_absolute_path(f"{config_path}/{config_name}.yaml")
        # save the config file to wandb from the hydras config
        wandb.save(config_file_path)

    if cfg.attack_type == 'pgd':
        eps_iters = [cfg.eps_iter] if type(cfg.eps_iter) == float else cfg.eps_iter
        for eps_iter in eps_iters:
            cfg.eps_iter = eps_iter
            for epsilon in epsilons:
                # the output directory should depend on the current directory and the checkpoint path and the attack type and epsilon
                output_dir = os.path.join(os.getcwd(), f"diffusion_policy/data/experiments/image/{task}/{algo}/eval_{checkpoint.split('/')[-3]}_{epsilon}_{view}")
                if os.path.exists(output_dir):
                    raise ValueError(f"Output path {output_dir} already exists!")

                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

                payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
                cfg_loaded = payload['cfg']
                cfg.action_space = cfg_loaded.shape_meta.action.shape

                cls = hydra.utils.get_class(cfg_loaded._target_)
                workspace = cls(cfg_loaded, output_dir=output_dir)
                workspace: BaseWorkspace
                workspace.load_payload(payload, exclude_keys=None, include_keys=None)
                try:
                    policy = workspace.model
                except AttributeError:
                    policy = workspace.policy

                if attack:
                    print("Running adversarial Attack")
                    cfg_loaded.task.env_runner['_target_'] = cfg._target_
                    cfg_loaded.task.env_runner['n_envs'] = n_envs
                if cfg.n_test > 0:
                    cfg_loaded.task.env_runner['n_test'] = cfg.n_test
                if cfg.n_train > 0:
                    cfg_loaded.task.env_runner['n_train'] = cfg.n_train

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
                    print(attack, env_runner)
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
                        "Epsilon":float(epsilon), "eps_iter": float(eps_iter)})
                print("Test/mean_score: ", json_log["test/mean_score"])
                out_path = os.path.join(output_dir, 'eval_log.json')
                json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
            wandb.finish()
    else:
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

            try:
                policy = workspace.model
            except AttributeError:
                policy = workspace.policy

            if attack:
                print("Running adversarial Attack")
                cfg_loaded.task.env_runner['_target_'] = cfg._target_
                cfg_loaded.task.env_runner['n_envs'] = n_envs
            if cfg.n_test > 0:
                cfg_loaded.task.env_runner['n_test'] = cfg.n_test
            if cfg.n_train > 0:
                cfg_loaded.task.env_runner['n_train'] = cfg.n_train
            cfg_loaded.task.env_runner['dataset_path'] = str(dataset_path)
            cfg.action_space = cfg_loaded.shape_meta.action.shape
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