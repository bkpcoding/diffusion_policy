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
import pickle
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_image_runner import AdversarialRobomimicImageRunner
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate
from hydra.core.global_hydra import GlobalHydra
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from pathlib import Path


def create_keypoint_animation(image_dataset, activations, output_path, object_coordinates = None):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        ax.imshow(image_dataset[frame].squeeze().permute(1, 2, 0).cpu().numpy())
        frame_activations = activations[frame].reshape(32, 2)
        frame_activations = (frame_activations + 1) / 2 * 75
        ax.scatter(frame_activations[:, 1], frame_activations[:, 0], c='r', s=20)
        if object_coordinates is not None:
            ax.scatter(object_coordinates[frame][0], object_coordinates[frame][1], c='g', s=20)
        ax.set_xlim(0, 75)
        ax.set_ylim(75, 0)
        ax.set_title(f"Frame {frame}")
        return ax

    anim = FuncAnimation(fig, update, frames=len(image_dataset), interval=1000)
    anim.save(output_path, writer='imagemagick', fps=1)
    plt.close(fig)

def apply_fgsm_attack(obs_dict, policy, cfg, predicted_action=None):
    """
    Applies the FGSM attack to the input image 
    """
    view = cfg.view
    if view == 'both':
        views = ['agentview_image', 'robot0_eye_in_hand_image']
    elif isinstance(view, list):
        views = view
    elif isinstance(view, str):
        views = [view]
    else:
        raise ValueError("view must be a string or a list of strings")
    # create a copy of the original dict that requires grad for backward pass
    obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
    policy.zero_grad()
    if predicted_action == None:
        with torch.no_grad():
            predicted_action = policy.predict_action(obs_dict)['action']
    batch = {}
    batch['obs'] = obs_dict
    batch['action'] = predicted_action
    loss, _ = policy.compute_loss(batch)
    print(loss.item())
    if cfg.targeted:
        loss = -loss
    loss.backward()
    if cfg.log:
        wandb.log({"loss":loss.item()})
    for view in views:
        grad = torch.sign(obs_dict[view].grad)
        if cfg.eps_iter != 'None':
            obs_dict[view] = obs_dict[view] + cfg.eps_iter * grad
        else:
            obs_dict[view] = obs_dict[view] + cfg.epsilon * grad
        obs_dict[view] = torch.clamp(obs_dict[view], cfg.clip_min, cfg.clip_max)
        if obs_dict[view].grad is not None:
            obs_dict[view].grad.zero_()
    return obs_dict


def apply_pgd_attack(obs_dict, policy, cfg):
    """
    Apply projected gradient descent attack from Madry et al. (2017)
    """
    view = cfg.view
    if view == 'both':
        views = ['agentview_image', 'robot0_eye_in_hand_image']
    elif isinstance(view, list):
        views = view
    elif isinstance(view, str):
        views = [view]
    else:
        raise ValueError("view must be a string or a list of strings")
    num_iter = cfg.num_iter
    clip_min = cfg.clip_min
    clip_max = cfg.clip_max
    norm = cfg.norm

    adv_obs_dict = obs_dict.copy()
    with torch.no_grad():
        predicted_action = policy.predict_action(obs_dict)['action']
        if cfg.targeted:
            predicted_action = predicted_action + torch.tensor(cfg.perturbations).to(obs_dict['agentview_image'].device)
    for i in range(num_iter):
        policy.zero_grad()
        adv_obs_dict = apply_fgsm_attack(adv_obs_dict, policy, cfg, predicted_action)
        for view in views:
            perturbation = adv_obs_dict[view] - obs_dict[view]
            if norm == 'l2':
                perturbation = perturbation * cfg.epsilon / torch.norm(perturbation, p=2)
            elif norm == 'linf':
                perturbation = torch.clamp(perturbation, -cfg.epsilon, cfg.epsilon)
            adv_obs_dict[view] = obs_dict[view] + perturbation
            adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], clip_min, clip_max)
    return adv_obs_dict

torch.backends.cudnn.enabled = False
@hydra.main(config_path='../interpretability_configs', config_name='bet_image_ph_pick')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_single_adversarial')
def main(cfg):
    checkpoint = cfg.checkpoint
    task = cfg.task
    device = cfg.device
    algo = cfg.algo

    # if cfg.log:
    #     wandb.init(project="diffusion_experimentation")

    # the output directory should depend on the current directory and the checkpoint path and the attack type and epsilon
    output_dir = os.path.join(os.getcwd(), f"diffusion_policy/data/experiments/image/{task}/{algo}/eval_single")
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

    try:
        if cfg_loaded.training.use_ema:
            policy = workspace.ema_model
    except:
        pass

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    cfg_loaded.task.env_runner['n_test'] = 1
    cfg_loaded.task.env_runner['n_train'] = 1
    cfg_loaded.task.env_runner['n_envs'] = 1
    env_runner = hydra.utils.instantiate(
        cfg_loaded.task.env_runner,
        output_dir=output_dir)
    env = env_runner.env
    obs = env.reset()
    action = np.array([[[1, 0, 0, 0, 0, 0, 0]]]).astype(np.float32)
    for i in range(10):
        obs, _, _, _ = env.step(action)
    obs = dict_apply(obs, lambda x: torch.tensor(x).to(device))
    view = 'robot0_eye_in_hand_image'
    original_obs = obs[view].clone()
    obs_dict = obs.copy()
    adv_obs_dict = apply_pgd_attack(obs_dict, policy, cfg)
    perturbed_obs = adv_obs_dict[view]
    obs_encoder = policy.obs_encoder
    image_encoder = obs_encoder.obs_nets['robot0_eye_in_hand_image']
    print(original_obs.squeeze(0)[1, :, 4:80, 4:80].unsqueeze(0).shape)
    original_image = original_obs.squeeze(0)[1, :, 4:80, 4:80].unsqueeze(0)
    perturbed_image = perturbed_obs.squeeze(0)[1, :, 4:80, 4:80].unsqueeze(0)
    original_activations = image_encoder(original_obs.squeeze(0)[1, :, 4:80, 4:80].unsqueeze(0)).detach().cpu().numpy()
    perturbed_activations = image_encoder(perturbed_obs.squeeze(0)[1, :, 4:80, 4:80].unsqueeze(0)).detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(original_image.squeeze().permute(1, 2, 0).cpu().numpy())
    original_activations = original_activations.reshape(32, 2)
    original_activations = (original_activations + 1) / 2 * 75
    ax.scatter(original_activations[:, 0], original_activations[:, 1], c='r', s=20)
    ax.set_xlim(0, 75)
    ax.set_ylim(75, 0)
    print(str(Path(__file__)))
    plt.savefig(str(Path(__file__)) + 'orginal_activations.png')

    fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(perturbed_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    perturbed_activations = perturbed_activations.reshape(32, 2)
    perturbed_activations = (perturbed_activations + 1) / 2 * 75
    ax.scatter(perturbed_activations[:, 0], perturbed_activations[:, 1], c='r', s=20)
    ax.set_xlim(0, 75)
    ax.set_ylim(75, 0)
    plt.savefig(str(Path(__file__)) + 'perturbed_activations.png')

    white_image = torch.zeros_like(original_image)
    white_image = white_image.to(device)
    white_image_activations = image_encoder(white_image).detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(white_image.squeeze().permute(1, 2, 0).cpu().numpy())
    white_image_activations = white_image_activations.reshape(32, 2)
    white_image_activations = (white_image_activations + 1) / 2 * 75
    ax.scatter(white_image_activations[:, 0], white_image_activations[:, 1], c='r', s=20)
    ax.set_xlim(0, 75)
    ax.set_ylim(75, 0)
    plt.savefig(str(Path(__file__)) + 'white_image_activations.png')

if __name__ == "__main__":
    main()