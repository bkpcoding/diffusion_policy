#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from baukit import Trace
import pickle
import os
import pathlib
import hydra
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from baukit import show
from omegaconf import OmegaConf
#%%
import matplotlib.pyplot as plt

def visualize_feature_maps(feature_maps, num_maps=6):
    fig, axes = plt.subplots(1, num_maps, figsize=(20, 5))
    for i, ax in enumerate(axes):
        if i < feature_maps.shape[1]:
            im = ax.imshow(feature_maps[0, i], cmap='viridis', aspect='auto', vmin=-5, vmax=5)
            ax.axis('off')
        else:
            ax.remove()
    # Create a colorbar with the specified range
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.show()

#%%
config_path = "/teamspace/studios/this_studio/bc_attacks/diffusion_policy/diffusion_policy/interpretability_configs/"
config_name = 'bet_image_ph_pick'
# config_name = 'lstm_gmm_image_ph_pick'
config_file = os.path.join(config_path, f"{config_name}.yaml")
cfg = OmegaConf.load(config_file)
checkpoint = cfg.checkpoint
task = cfg.task
device = cfg.device
algo = cfg.algo

# if cfg.log:
#     wandb.init(project="diffusion_experimentation")

# the output directory should depend on the current directory and the checkpoint path and the attack type and epsilon
output_dir = os.path.join(os.getcwd(), f"diffusion_policy/data/experiments/image/{task}/{algo}/eval_single")
if os.path.exists(output_dir):
    pass

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
#%%
# load the patch
try:
    patch = np.load(cfg.patch_path, allow_pickle=True)
except:
    patch = torch.load(cfg.patch_path)
patch = patch.squeeze(0)
patch = torch.zeros_like(patch)

#%%
# load the observation dicts from the pickle file
clean_obs = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/clean_obs_dicts_0.0625.pkl'
obs_dicts = pickle.load(open(clean_obs, 'rb'))
print(obs_dicts[0]['robot0_eye_in_hand_image'].shape)

#%%
# clean action prediction
clean_action_pred = policy.predict_action(obs_dicts[5], return_latent=True)['latent']
print(clean_action_pred[0][0, 0], clean_action_pred[1][0, 0])
clean_action = policy.predict_action(obs_dicts[5])['action']
print(clean_action[0][0])

#%%
perturbed_obs = obs_dicts[5].copy()
perturbed_obs['robot0_eye_in_hand_image'] = perturbed_obs['robot0_eye_in_hand_image'] + patch
perturbed_action_pred = policy.predict_action(perturbed_obs, return_latent=True)['latent']
print(perturbed_action_pred[0][0, 0], perturbed_action_pred[1][0, 0])
perturbed_action = policy.predict_action(perturbed_obs)['action']
print(perturbed_action[0][0])

#%%
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
if len(patch.shape) == 5:
    patch = patch[0][1]
print(f"Patch shape: {patch.shape}, image shape: {image_dataset[0].shape}")
print(f"L2 norm of the patch: {np.linalg.norm(patch)}")
