import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import dill
import wandb
import tqdm
import numpy as np
import shutil
import collections
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.train_robomimic_image_workspace import TrainRobomimicImageWorkspace
from diffusion_policy.utils.attack_utils import transform_square_patch
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly
import pickle

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils


checkpoint = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/ibc_dfo/train_2/checkpoints/epoch=2100-test_mean_score=1.000.ckpt'
payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
cfg_loaded = payload['cfg']
cls = hydra.utils.get_class(cfg_loaded._target_)
workspace = cls(cfg_loaded)
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
policy = workspace.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)
policy.eval()


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    print(env_meta)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

def plot_energy_landscape(policy, obs, device, timestep=0, grid_actions = None):
    np_obs_dict = dict(obs)
    obs_dict = dict_apply(np_obs_dict, lambda x: torch.tensor(x).unsqueeze(0).to(device))
    nobs = policy.normalizer.normalize(obs_dict)
    value = next(iter(nobs.values()))
    B, To = value.shape[:2]
    this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    nobs_features = policy.obs_encoder(this_nobs)
    nobs_features = nobs_features.reshape(B, To, -1)
    action = policy.predict_action(obs_dict, return_energy=True)['action'].cpu().numpy()
    
    # Extract the initial action (shape (7,))
    initial_action = action[0, 0, :]

    # Define the perturbation range and step size
    perturb_range = 0.5
    step_size = 0.01

    # Create the perturbations for the first two dimensions
    perturbations = np.arange(-perturb_range, perturb_range + step_size, step_size)
    grid_x, grid_y = np.meshgrid(perturbations, perturbations)

    if grid_actions is None:
        # Initialize an array to hold the grid of actions (shape (101, 101, 7))
        grid_actions = np.zeros((grid_x.shape[0], grid_y.shape[1], 7))

        # Fill the grid with actions
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                grid_actions[i, j, :2] = initial_action[:2] + np.array([grid_x[i, j], grid_y[i, j]])
                grid_actions[i, j, 2:] = initial_action[2:]

        # Reshape the grid to have shape (10201, 7)
        grid_actions = grid_actions.reshape(-1, 7)

        # Convert grid_actions to torch tensor
        grid_actions = torch.tensor(grid_actions, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2)

    energies = policy.forward(nobs_features, grid_actions)
    energies_np = energies.cpu().detach().numpy()

    # Flatten the grid for plotting
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()
    energies_flat = energies_np.flatten()
    
    return grid_x_flat, grid_y_flat, energies_flat


def create_animation(frames, output_file):
    layout = go.Layout(
        title='Energy Distribution in Action Space Over Time',
        scene=dict(
            xaxis=dict(title='Dimension 1 Perturbation'),
            yaxis=dict(title='Dimension 2 Perturbation'),
            zaxis=dict(title='Energy')
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'showactive': False,
            'type': 'buttons'
        }]
    )

    fig = go.Figure(data=frames[0]['data'], layout=layout, frames=frames)

    # Save the plot as an HTML file
    pyo.plot(fig, filename=output_file)

def plot_from_env():

    cfg_loaded.task.env_runner['_target_'] = 'diffusion_policy.env_runner.robomimic_single_image_runner.RobomimicSingleImageRunner'
    cfg_loaded.task.env_runner.dataset_path = os.path.join('/teamspace/studios/this_studio/bc_attacks/diffusion_policy', cfg_loaded.task.dataset_path)
    env_runner = hydra.utils.instantiate(
        cfg_loaded.task.env_runner,
        output_dir=None
    )
    env = env_runner.env
    env.reset()
    max_timesteps = 60
    # collect observations and actions for the policy for one rollout
    import tqdm
    obs = env.reset()
    done = False
    with tqdm.tqdm(total=max_timesteps, desc="Timesteps") as pbar:
        timestep = 0
        while not done and timestep < max_timesteps:
            print(obs['agentview_image'].shape, obs['robot0_eye_in_hand_image'].shape)
            plot_energy_landscape(policy, obs, device, timestep=timestep)
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.tensor(x).unsqueeze(0).to(device))
            action = policy.predict_action(obs_dict)['action'].squeeze(0).cpu().numpy()
            obs, _, done, _ = env.step(action)
            timestep += 1
            pbar.update(1)
    env.close()


def plot_from_file(policy, device, filename, output_file):
    observations = pickle.load(open(filename, 'rb'))
    observations = [observations]
    frames = []
    obs = observations[0]
    # obs = observations['0']
    np_obs_dict = dict(obs)
    obs_dict = dict_apply(np_obs_dict, lambda x: torch.tensor(x).to(device))
    nobs = policy.normalizer.normalize(obs_dict)
    value = next(iter(nobs.values()))
    B, To = value.shape[:2]
    this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    nobs_features = policy.obs_encoder(this_nobs)
    nobs_features = nobs_features.reshape(B, To, -1)
    print("Obs dict shape: ", obs_dict['agentview_image'].shape)
    action = policy.predict_action(obs_dict, return_energy=True)['action'].cpu().numpy()
    
    # Extract the initial action (shape (7,))
    initial_action = action[0, 0, :]

    # Define the perturbation range and step size
    perturb_range = 0.5
    step_size = 0.01

    # Create the perturbations for the first two dimensions
    perturbations = np.arange(-perturb_range, perturb_range + step_size, step_size)
    grid_x, grid_y = np.meshgrid(perturbations, perturbations)
    # Initialize an array to hold the grid of actions (shape (101, 101, 7))
    grid_actions = np.zeros((grid_x.shape[0], grid_y.shape[1], 7))

    # Fill the grid with actions
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            grid_actions[i, j, :2] = initial_action[:2] + np.array([grid_x[i, j], grid_y[i, j]])
            grid_actions[i, j, 2:] = initial_action[2:]

    # Reshape the grid to have shape (10201, 7)
    grid_actions = grid_actions.reshape(-1, 7)

    # Convert grid_actions to torch tensor
    grid_actions = torch.tensor(grid_actions, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2)    
    
    for timestep, observation in enumerate(tqdm.tqdm(observations, desc="Timesteps")):
        observation = dict(observation)
        # observation = dict_apply(observation, lambda x: torch.tensor(x).squeeze(0).to(device))
        observation = dict_apply(observation, lambda x: torch.tensor(x)[0].to(device))
        print(observation['agentview_image'].shape, observation['robot0_eye_in_hand_image'].shape)
        grid_x_flat, grid_y_flat, energies_flat = plot_energy_landscape(policy, observation, device, timestep, grid_actions)
        
        # Create a trace for this frame
        trace = go.Scatter3d(
            x=grid_x_flat,
            y=grid_y_flat,
            z=energies_flat,
            mode='markers',
            marker=dict(
                size=5,
                color=energies_flat,
                colorscale='Viridis',
                colorbar=dict(title='Energy')
            )
        )
        
        frames.append(go.Frame(data=[trace], name=f'Timestep {timestep}'))
    
    create_animation(frames, output_file)


def plot_hist(clean_energy, perturbed_energy):
    # plot a histogram of the energy values
    clean_energy = pickle.load(open(clean_energy, 'rb'))[:10]
    perturbed_energy = pickle.load(open(perturbed_energy, 'rb'))[:10]
    clean_energy = clean_energy.reshape(-1)
    perturbed_energy = perturbed_energy.reshape(-1)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean_energy, name='Clean Energy'))
    fig.add_trace(go.Histogram(x=perturbed_energy, name='Perturbed Energy'))
    fig.update_layout(title='Histogram of Energy Values', xaxis_title='Energy Value', yaxis_title='Frequency')
    # make the background white
    fig.update_layout(plot_bgcolor='white')
    fig.write_html('/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/energy_landscape/energy_histogram_10.html')

def plot_2d_heatmap(clean_obs, perturbed_obs):
    clean_dict = pickle.load(open(clean_obs, 'rb'))
    perturbed_dict = pickle.load(open(perturbed_obs, 'rb'))
    clean_dict = dict(clean_dict)
    perturbed_dict = dict(perturbed_dict)
    clean_dict = dict_apply(clean_dict, lambda x: torch.tensor(x)[0].unsqueeze(0).to(device))
    perturbed_dict = dict_apply(perturbed_dict, lambda x: torch.tensor(x)[0].unsqueeze(0).to(device))

    # initial action of clean and perturbed observations
    clean_action = policy.predict_action(clean_dict)['action'].cpu().numpy()
    perturbed_action = policy.predict_action(perturbed_dict)['action'].cpu().numpy()

    clean_dict = dict_apply(clean_dict, lambda x: torch.tensor(x).squeeze(0).to(device))
    perturbed_dict = dict_apply(perturbed_dict, lambda x: torch.tensor(x).squeeze(0).to(device))

    initial_clean_action = clean_action[0, 0, :]
    initial_perturbed_action = perturbed_action[0, 0, :]

    # Define the perturbation range and step size
    perturb_range = 0.5
    step_size = 0.01

    # Create the perturbations for the first two dimensions
    perturbations = np.arange(-perturb_range, perturb_range + step_size, step_size)
    grid_x, grid_y = np.meshgrid(perturbations, perturbations)

    # Initialize an array to hold the grid of actions (shape (101, 101, 7))
    grid_clean_actions = np.zeros((grid_x.shape[0], grid_y.shape[1], 7))
    grid_perturbed_actions = np.zeros((grid_x.shape[0], grid_y.shape[1], 7))

    # Fill the grid with actions
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            grid_clean_actions[i, j, :2] = initial_clean_action[:2] + np.array([grid_x[i, j], grid_y[i, j]])
            grid_clean_actions[i, j, 2:] = initial_clean_action[2:]
            grid_perturbed_actions[i, j, :2] = initial_perturbed_action[:2] + np.array([grid_x[i, j], grid_y[i, j]])
            grid_perturbed_actions[i, j, 2:] = initial_perturbed_action[2:]
    
    # Reshape the grid to have shape (10201, 7)
    grid_clean_actions = grid_clean_actions.reshape(-1, 7)
    grid_perturbed_actions = grid_perturbed_actions.reshape(-1, 7)

    # Convert grid_actions to torch tensor
    grid_clean_actions = torch.tensor(grid_clean_actions, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2)
    grid_perturbed_actions = torch.tensor(grid_perturbed_actions, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2)

    print("Clean dict shape: ", clean_dict['agentview_image'].shape)
    clean_grid_x, clean_grid_y, clean_energies = plot_energy_landscape(policy, clean_dict, device, grid_actions=grid_clean_actions)
    perturbed_grid_x, perturbed_grid_y, perturbed_energies = plot_energy_landscape(policy, perturbed_dict, device, grid_actions=grid_perturbed_actions)
    clean_energies = clean_energies.reshape(101, 101)
    perturbed_energies = perturbed_energies.reshape(101, 101)
    # Create the 2D heatmap of the energy landscape for 2 subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Clean Energy Landscape', 'Perturbed Energy Landscape'))
    fig.add_trace(go.Heatmap(x=clean_grid_x, y=clean_grid_y, z=clean_energies, colorscale='Viridis', name='Clean Energy Landscape'), row=1, col=1)
    fig.add_trace(go.Heatmap(x=perturbed_grid_x, y=perturbed_grid_y, z=perturbed_energies, colorscale='Viridis', name='Perturbed Energy Landscape'), row=1, col=2)
    fig.update_layout(title='2D Heatmap of Energy Landscape', xaxis_title='Dimension 1 Perturbation', yaxis_title='Dimension 2 Perturbation')
    fig.write_html('/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/energy_landscape/2d_heatmap.html')

if __name__ == '__main__':
    perturbed_obs = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/ibc_perturbed_obs_dict_0.0625_timestep_9.pkl'
    clean_obs = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/ibc_clean_obs_dict_0.0625_timestep_9.pkl'
    clean_energy = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/ibc_clean_energy_0.0625_timestep_8.pkl'
    perturbed_energy = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/ibc_perturbed_energy_0.0625_timestep_8.pkl'
    output_file = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/energy_landscape/perturbed_energy_landscape_animation.html'
    
    # Define your policy and device here
    # policy = ...
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = clean_obs
    # plot_from_file(policy, device, filename, output_file)
    # plot_2d_heatmap(clean_obs, perturbed_obs)
    plot_hist(clean_energy, perturbed_energy)
