import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def lipschitz_video(lipschitz_values_path:str, obs_dicts_path:str):
    # open the pickle files
    if lipschitz_values_path is not None:
        with open(lipschitz_values_path, 'rb') as f:
            lipschitz_values = CPU_Unpickler(f).load()
    with open(obs_dicts_path, 'rb') as f:
        obs_dicts = CPU_Unpickler(f).load()
    # print(len(lipschitz_values), len(obs_dicts))
    # print(lipschitz_values[0], obs_dicts[0])
    # video using obs_dicts 'agentview_image'
    ims = []
    fig, ax = plt.subplots()
    ax.axis('off')
    for i in range(len(obs_dicts)):
        im = ax.imshow(obs_dicts[i]['agentview_image'][2, 0, :, :, :].permute(1, 2, 0).detach().numpy())
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    # ani.save('./videos/clean_02_robot_eye_in_hand_randtar_True.gif', writer='imagemagick', fps=4)
    ani.save('./videos/diffusion_policy_perturbed_agentview_3.gif', writer='imagemagick', fps=4)

# lipschitz_values_path = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/lipschitz_consts.pkl'
lipschitz_values_path = None
obs_dicts_path = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/diffusion_policy_observed_dicts.pkl'
lipschitz_video(lipschitz_values_path, obs_dicts_path)