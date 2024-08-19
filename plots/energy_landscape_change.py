import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from matplotlib import animation

energy_landscape = "/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/energy_landscape/energy_0.0625_9.pkl"

# Load the energy landscape
with open(energy_landscape, 'rb') as f:
    energy = pickle.load(f)

print(len(energy), energy[0].shape, energy[0].max(), energy[0].min(), energy[1].shape, energy[1].max(), energy[1].min())

