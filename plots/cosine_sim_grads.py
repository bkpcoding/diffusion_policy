import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


grad1 = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/vanilla_bc/train_0/checkpoints/gradients_untar_pert_0.0625.pkl'
grad2 = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/bet/train6/checkpoints/gradients_untar_pert_0.0625.pkl'

with open(grad1, 'rb') as f:
    grad1 = pkl.load(f)
with open(grad2, 'rb') as f:
    grad2 = pkl.load(f)

for key in grad1.keys():
    grad1[key] = grad1[key].squeeze(0).cpu().numpy()
    grad2[key] = grad2[key].squeeze(0)[0].cpu().numpy()

cosine_sim = []
for key in grad1.keys():
    cosine_sim.append(np.dot(grad1[key].flatten(), grad2[key].flatten()) / (np.linalg.norm(grad1[key]) * np.linalg.norm(grad2[key]) + 1e-8))
cosine_sim = np.array(cosine_sim)

plt.plot(cosine_sim)
plt.savefig('cosine_sim.png')
