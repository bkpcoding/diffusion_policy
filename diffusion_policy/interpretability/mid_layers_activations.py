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
# config_name = 'bet_image_ph_pick'
config_name = 'lstm_gmm_image_ph_pick'
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
net = image_encoder.backbone.nets


#%%
with Trace(net, '6.1.relu') as ret:
    _ = net(image_dataset[0])
    representation = ret.output
representation_clean = representation.detach().cpu().numpy()
print(f"Representation shape: {representation_clean.shape}")
visualize_feature_maps(representation_clean, num_maps=6)

# get the activations for the image dataset
# image_activations = []
# perturbed_activations = []
# no_red_activations = []
# for image in image_dataset:
#     image_activations.append(image_encoder(image).detach().cpu().numpy())
# for image in perturbed_image_dataset:
#     perturbed_activations.append(image_encoder(image).detach().cpu().numpy())
# for image in no_red_image_dataset:
#     no_red_activations.append(image_encoder(image).detach().cpu().numpy())



# %%
with Trace(net, '6.1.relu') as ret:
    _ = net(perturbed_image_dataset[0])
    representation = ret.output
representation_perturbed = representation.detach().cpu().numpy()
print(f"Representation shape: {representation_perturbed.shape}")
visualize_feature_maps(representation_perturbed, num_maps=6)

# %%
# plot the difference between the clean and perturbed representations
diff = representation_clean - representation_perturbed
print(diff[0])
visualize_feature_maps(diff, num_maps=6)
# %%
# get the activations of some layers before
