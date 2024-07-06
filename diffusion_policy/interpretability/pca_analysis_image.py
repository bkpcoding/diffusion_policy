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
    object_dataset = object_dataset[:, 7:9]
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
        perturbed_image_dataset.append(np.concatenate([image, patch], axis=0))
    # convert the images to tensor
    image_dataset = [torch.tensor(image)[0].to(device).unsqueeze(0) for image in image_dataset]
    image_dataset = [image[:, :, 4:80, 4:80] for image in image_dataset]
    perturbed_image_dataset = [torch.tensor(image)[0].to(device).unsqueeze(0) for image in perturbed_image_dataset]
    perturbed_image_dataset = [image[:, :, 4:80, 4:80] for image in perturbed_image_dataset]
    no_red_image_dataset = [torch.tensor(image).to(device).unsqueeze(0) for image in no_red_image_dataset]
    no_red_image_dataset = [image[:, :, 4:80, 4:80] for image in no_red_image_dataset]
    # image_dataset = image_dataset[10:30]
    # no_red_image_dataset = no_red_image_dataset[10:30]
    # perturbed_image_dataset = perturbed_image_dataset[10:30]
    # object_dataset = object_dataset[10:30]

    activations = []
    perturbed_activations = []
    for image in image_dataset:
        activations.append(image_encoder(image).detach().cpu().numpy())
        # print activations shape of image_encoder.backbone output
    
    for image in perturbed_image_dataset:
        perturbed_activations.append(image_encoder(image).detach().cpu().numpy())
    linear_probing_mse, model = linear_probing(activations, object_dataset)
    print(f"Linear Probing MSE: {linear_probing_mse}")

    no_red_activations = []
    for image in no_red_image_dataset:
        no_red_activations.append(image_encoder(image).detach().cpu().numpy())
    # no_red_linear_probing_mse, no_red_model = linear_probing(no_red_activations, object_dataset)
    # print(f"No red linear probing MSE: {no_red_linear_probing_mse}")
    no_red_object_dataset = model.predict(np.array(no_red_activations).reshape(len(no_red_activations), -1))
    # calculate the mean squared error between the no red object dataset and the original object dataset
    mse_no_red = mean_squared_error(object_dataset[:, 7:9], no_red_object_dataset)
    print("Mean Squared Error between no red and original object dataset: ", mse_no_red)
    # apply the model to the perturbed activations
    perturbed_object_dataset = model.predict(np.array(perturbed_activations).reshape(len(perturbed_activations), -1))
    # calculate the mean squared error between the perturbed object dataset and the original object dataset
    mse = mean_squared_error(object_dataset[:, 7:9], perturbed_object_dataset)
    print(f"Mean Squared Error between original and perturbed object dataset: {mse}")
    # plot the 3D plot of the object dataset and the perturbed object dataset on the same plot
    import plotly.express as px
    import plotly.graph_objects as go
    # Create traces
    trace1 = go.Scatter(
        x=object_dataset[:, 0],
        y=object_dataset[:, 1],
        mode='markers',
        marker=dict(size=5),
        name='original'
    )

    trace2 = go.Scatter(
        x=perturbed_object_dataset[:, 0],
        y=perturbed_object_dataset[:, 1],
        mode='markers',
        marker=dict(size=5),
        name='perturbed'
    )
    trace3 = go.Scatter(
        x=no_red_object_dataset[:, 0],
        y=no_red_object_dataset[:, 1],
        mode='markers',
        marker=dict(size=5, opacity=0.5),
        name='no red'
    )

    # Create the layout
    layout = go.Layout(
        title='Object and Perturbed Dataset and No Red Dataset',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
        )
    )
    # Create the figure
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    # Save the figure
    fig.write_html(os.path.join(cfg.plot_path, f'object_and_perturbed_no_red_samemodel_rel_dataset_300pts_tar_{algo}.html'))

    # image_dataset = image_dataset[:30]
    # object_dataset = object_dataset[:30]
    # do linear probing on the image dataset
    # plot the images as gif to see the dataset
    # from matplotlib import animation
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ims = []
    # for image in no_red_image_dataset:
    #     image = image.transpose(1, 2).transpose(2, 3)
    #     ims.append([plt.imshow(image[0].cpu().numpy(), animated=True)])
    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    # ani.save(os.path.join(cfg.plot_path, 'no_red_image_dataset.gif'))
    # # crop the images to 76x76 from 84x84
    # image_dataset = [image[:, :, 4:80, 4:80] for image in image_dataset]
    # activations = []
    # for image in image_dataset:
    #     activations.append(image_encoder(image).detach().cpu().numpy())
    # # do PCA analysis on the activations
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=3)
    # activations = np.array(activations)
    # activations = activations.reshape(activations.shape[0], -1)
    # print(activations.shape)
    # pca.fit(activations)
    # print(pca.explained_variance_ratio_)
    # # plot the PCA analysis
    # import plotly.express as px
    # # plot the PCA analysis
    # # fig = px.scatter(x=activations[:, 0], y=activations[:, 1])
    # # color the points by the index
    # fig = px.scatter_3d(x=activations[:, 0], y=activations[:, 1], z=activations[:, 2], color=range(activations.shape[0]), opacity=0.5, title='PCA Analysis of Image Encoder Activations')
    # fig.write_html(os.path.join(cfg.plot_path, 'pca_analysis.html'))
    # # plot the 3D plot of the object dataset
    # fig = px.scatter_3d(x=object_dataset[:, 7], y=object_dataset[:, 8], z=object_dataset[:, 9], color=range(object_dataset.shape[0]), opacity=0.5, title='Object Dataset')
    # fig.write_html(os.path.join(cfg.plot_path, 'object_dataset.html'))

if __name__ == '__main__':
    main()
