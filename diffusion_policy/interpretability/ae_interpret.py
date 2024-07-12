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
from diffusion_policy.interpretability.autoencoder import Autoencoder
from diffusion_policy.interpretability.autoencoder_loss import autoencoder_loss


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
# @hydra.main(config_path='../interpretability_configs', config_name='vanilla_bc_image_ph_pick')
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
    try:
        patch = np.load(cfg.patch_path, allow_pickle=True)
    except:
        patch = torch.load(cfg.patch_path)
    if type(patch) == torch.Tensor:
        patch = patch.cpu().numpy()
    print(f"Patch shape: {patch.shape}, image shape: {image_dataset[0].shape}")
    print(f"L2 norm of the patch: {np.linalg.norm(patch)}")
    # apply the patch to the image dataset
    perturbed_image_dataset = []
    for image in image_dataset:
        try:
            perturbed_image_dataset.append(image + patch)
        except ValueError:
            patch = np.expand_dims(patch, axis=0)
            perturbed_image_dataset.append(image + patch)
            print(np.concatenate([image, patch], axis=0).shape)
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
 
    # train the autoencoder
    autoencoder = Autoencoder(2, 64, torch.nn.Identity())
    # convert the image dataset to a tensor with training and testing data
    image_activations = np.array(image_activations)
    image_activations = image_activations.reshape(image_activations.shape[0], -1)
    X_train, X_test, _, _ = train_test_split(image_activations, image_activations, test_size=0.2)
    # convert into a tensor
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    epochs = 1000
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)
    for epoch in range(epochs):
        latents_pre_act, latents, recons = autoencoder(X_train)
        loss = autoencoder_loss(recons, X_train, latents_pre_act, 0.2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"Epoch {epoch}, loss: {loss.item()}")
    
    # calculate the accuracy of the autoencoder
    X_test_latent, _ = autoencoder.encode(X_test)
    X_test_recons = autoencoder.decode(X_test_latent)
    test_loss = torch.nn.functional.mse_loss(X_test_recons, X_test)
    print(f"Test loss: {test_loss.item()}")

    # check the reconstruction loss on perturbed images
    perturbed_activations = torch.tensor(perturbed_activations).float()
    perturbed_latents, _ = autoencoder.encode(perturbed_activations)
    perturbed_recons = autoencoder.decode(perturbed_latents)
    perturbed_loss = torch.nn.functional.mse_loss(perturbed_recons, perturbed_activations)
    print(f"Perturbed loss: {perturbed_loss.item()}")

    no_red_activations= torch.tensor(no_red_activations).float()
    no_red_latents, _ = autoencoder.encode(no_red_activations)
    no_red_recons = autoencoder.decode(no_red_latents)
    no_red_loss = torch.nn.functional.mse_loss(no_red_recons, no_red_activations)
    print(f"No red loss: {no_red_loss.item()}")

    # plot the latent space of both the image and perturbed image dataset
    image_latents, _ = autoencoder.encode(torch.tensor(image_activations).float())
    image_latents = image_latents.detach().cpu().numpy().reshape(-1, 2)[0:100]
    perturbed_latents = perturbed_latents.detach().cpu().numpy().reshape(-1, 2)[0:100]
    no_red_latents = no_red_latents.detach().cpu().numpy().reshape(-1, 2)[0:100]
    plt.scatter(image_latents[:, 0], image_latents[:, 1], c='r', label='Image dataset')
    plt.scatter(perturbed_latents[:, 0], perturbed_latents[:, 1], c='b', label='Perturbed image dataset')
    plt.scatter(no_red_latents[:, 0], no_red_latents[:, 1], c='g', label='No red image dataset')
    plt.legend()
    plt.savefig('/teamspace/studios/this_studio/bc_attacks/diffusion_policy/diffusion_policy/interpretability/latent_space_train0_untar0625_bet_smoothed.png')


if __name__ == '__main__':
    main()

