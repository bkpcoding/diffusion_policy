# plot the energy function of the samples

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from matplotlib import animation


# load the samplesxlogits from samplesxlogits.pkl
samplesxlogits = pickle.load(open('samplesxlogits.pkl', 'rb'))
# convert the samplesxlogits to a numpy array from tensor
obs = np.load('obs_dict.npy', allow_pickle=True)
energy_values = []
for keys, values in samplesxlogits.items():
    energy_values.append(values[1].detach().numpy())
energy_values = np.array(energy_values)
print(energy_values.shape)
print(energy_values[0])

x = np.arange(0, energy_values.shape[-1], 1)
# plot the energy function of the samples
for i in range(energy_values.shape[0]):
    # draw dots for the values of the energy function
    plt.scatter(x, energy_values[i, 0, :])
    plt.savefig(f'energy_function_{i}.png')
# animate the change in the energy function with time
fig = plt.figure()
ims = []
for i in range(energy_values.shape[0]):
    im = plt.plot(x, energy_values[i, 0, :])
    ims.append(im)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save('energy_function.gif', writer='imagemagick', fps=2)