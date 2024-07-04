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

def linear_probing(activations, object_dataset):
    # convert to numpy array
    activations = np.array(activations)
    activations = activations.reshape(activations.shape[0], -1)
    object_dataset = np.array(object_dataset)
    object_dataset = object_dataset[:, 7:10]
    print(activations.shape, object_dataset.shape)
    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(activations, object_dataset, test_size=0.2)
    # fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # predict the test set
    y_pred = model.predict(X_test)
    # calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    return mse, model


torch.backends.cudnn.enabled = False
@hydra.main(config_path='../interpretability_configs', config_name='lstm_gmm_image_ph_pick')
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
    policy = workspace.model

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
    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    image_encoder = obs_encoder.obs_nets['robot0_eye_in_hand_image']
    # convert the images to tensor
    image_dataset = [torch.tensor(image)[0].to(device).unsqueeze(0) for image in image_dataset]
    image_dataset = [image[:, :, 4:80, 4:80] for image in image_dataset]
    no_red_image_dataset = [torch.tensor(image).to(device).unsqueeze(0) for image in no_red_image_dataset]
    no_red_image_dataset = [image[:, :, 4:80, 4:80] for image in no_red_image_dataset]
    image_dataset = image_dataset[10:30]
    no_red_image_dataset = no_red_image_dataset[10:30]
    object_dataset = object_dataset[10:30]

    activations = []
    for image in image_dataset:
        activations.append(image_encoder(image).detach().cpu().numpy())
    linear_probing_mse, model = linear_probing(activations, object_dataset)
    print(f"Linear probing MSE: {linear_probing_mse}")

    no_red_activations = []
    for image in no_red_image_dataset:
        no_red_activations.append(image_encoder(image).detach().cpu().numpy())
    no_red_linear_probing_mse, no_red_model = linear_probing(no_red_activations, object_dataset)
    print(f"No red linear probing MSE: {no_red_linear_probing_mse}")

if __name__ == '__main__':
    main()