import numpy as np
import matplotlib.pyplot as plt
import wandb
import pickle


def plot_loss_per_iter(filename: str):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    plt.figure()
    for i in range(100):
        plt.plot(data[i], label=f'iter {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/images/loss_per_episode_0.125_randtar_False_linf_lr_0.005_same_action_samples.png')

filename = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/loss_per_episode_0.125_randtar_False_linf_lr_0.005_same_action_samples.pkl'
lipschitz_values = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/loss_per_episode_0.125_randtar_False_linf_lr_0.005.pkl'

plot_loss_per_iter(filename)