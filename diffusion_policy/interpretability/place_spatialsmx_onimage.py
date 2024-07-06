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


torch.backends.cudnn.enabled = False
@hydra.main(config_path='../interpretability_configs', config_name='bet_image_ph_pick')
# @hydra.main(config_path='../interpretability_configs', config_name='lstm_gmm_image_ph_pick')
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

    # load the dataset from config.image_dataset
    image_dataset = pickle.load(open(cfg.image_dataset, 'rb'))
    object_dataset = pickle.load(open(cfg.object_dataset, 'rb'))
    no_red_image_dataset = pickle.load(open(cfg.no_red_image_dataset, 'rb'))
    for i in range(len(no_red_image_dataset)):
        image = no_red_image_dataset[i]
        if len(image.shape) != 3:
            # remove the image from the dataset
            no_red_image_dataset.remove(image)
        if image.shape[0] != 3:
            no_red_image_dataset[i] = image.transpose(2, 0, 1)
 
    # convert object_dataset into a numpy array
    object_dataset = np.array(object_dataset)
    try:
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    except:
        obs_encoder = policy.obs_encoder
    image_encoder = obs_encoder.obs_nets['robot0_eye_in_hand_image']
    # load the universal patch
    patch = np.load(cfg.patch_path, allow_pickle=True)
    print(f"Patch shape: {patch.shape}, image shape: {image_dataset[0].shape}")
    if type(patch) == torch.Tensor:
        patch = patch.cpu().numpy()
    # apply the patch to the image dataset
    perturbed_image_dataset = []
    for image in image_dataset:
        try:
            perturbed_image_dataset.append(image + patch)
        except ValueError:
            patch = np.expand_dims(patch, axis=0)
            perturbed_image_dataset.append(image + patch)
    # convert the images to tensor
    image_dataset = [torch.tensor(image)[0].to(device).unsqueeze(0) for image in image_dataset]
    image_dataset = [image[:, :, 4:80, 4:80] for image in image_dataset]
    perturbed_image_dataset = [torch.tensor(image)[0].to(device).unsqueeze(0) for image in perturbed_image_dataset]
    perturbed_image_dataset = [image[:, :, 4:80, 4:80] for image in perturbed_image_dataset]
    no_red_image_dataset = [torch.tensor(image).to(device).unsqueeze(0) for image in no_red_image_dataset]
    no_red_image_dataset = [image[:, :, 4:80, 4:80] for image in no_red_image_dataset]
    print(f"Length of image and perturbed image dataset: {len(image_dataset)}, {image_dataset[0].shape}")

    # get the activations for the image dataset
    image_activations = []
    perturbed_activations = []
    no_red_activations = []
    for image in image_dataset:
        image_activations.append(image_encoder(image).detach().cpu().numpy())
    for image in perturbed_image_dataset:
        perturbed_activations.append(image_encoder(image).detach().cpu().numpy())
    for image in no_red_image_dataset:
        no_red_activations.append(image_encoder(image).detach().cpu().numpy())
    object_xy_coords = object_dataset[:, 7:9]
    object_xy_coords = (object_xy_coords + 1) / 2 * 75
    
    datasets = {'image': image_dataset, 'perturbed_image': perturbed_image_dataset, 'no_red_image': no_red_image_dataset}
    activations = {'image': image_activations, 'perturbed_image': perturbed_activations, 'no_red_image': no_red_activations}
    # datasets = {'perturbed_image': perturbed_image_dataset}
    for dataset_name, dataset in datasets.items():
        output_path = f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/diffusion_policy/interpretability/keypoints_on_{dataset_name}_lstm.gif'
        create_keypoint_animation(dataset, activations[dataset_name], output_path, object_xy_coords)
        print(f"Animation for {dataset_name} saved to {output_path}")



if __name__ == '__main__':
    main()

