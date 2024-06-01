import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer

class DiffusionUnetLowdimPolicy2D(DiffusionUnetLowdimPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = 2
        self.obs_dim = 1
        self.num_inference_steps = 20
        self.obs_as_global_cond = True
        self.pred_action_steps_only = True
        self.n_action_steps = 1

    def generate_data(self, num_samples):
        # Generate data with three modes in the action space
        modes = [0.25, 0.5, 0.75]
        obs = np.random.choice(modes, size=(num_samples, 1))
        actions = np.zeros((num_samples, self.n_action_steps, self.action_dim))
        for i in range(num_samples):
            mode = obs[i, 0]
            actions[i] = np.random.normal(loc=mode, scale=0.1, size=(self.n_action_steps, self.action_dim))
        actions = np.clip(actions, 0, 1)
        return {'obs': obs, 'action': actions}

    def animate_sampling(self, obs):
        device = self.device
        dtype = self.dtype
        B = 1
        T = self.n_action_steps
        Da = self.action_dim
        global_cond = torch.from_numpy(obs).reshape(1, -1).to(device).to(dtype)
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        trajectory = torch.randn(size=shape, device=device, dtype=dtype)
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        fig, ax = plt.subplots()
        scatter = ax.scatter([], [])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Action Dim 1')
        ax.set_ylabel('Action Dim 2')
        title = ax.text(0.5, 1.01, '', transform=ax.transAxes, ha='center')

        def update(t):
            trajectory[cond_mask] = cond_data[cond_mask]
            model_output = self.model(trajectory, t, global_cond=global_cond)
            trajectory[:] = scheduler.step(model_output, t, trajectory).prev_sample
            scatter.set_offsets(trajectory[0].cpu().numpy())
            title.set_text(f'Timestep: {t}')
            return scatter, title

        ani = FuncAnimation(fig, update, frames=scheduler.timesteps, blit=True)
        plt.close(fig)
        return ani

def train_policy(policy, data, num_epochs, batch_size, lr):
    # Create a DataLoader from the generated data
    obs = torch.from_numpy(data['obs']).float()
    actions = torch.from_numpy(data['action']).float()
    dataset = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(policy.device)
            batch_actions = batch_actions.to(policy.device)
            batch = {'obs': batch_obs, 'action': batch_actions}

            # Forward pass and loss computation
            loss = policy.compute_loss(batch)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

model = ConditionalUnet1D(input_dim=2, global_cond_dim=1)
noise_scheduler = DDPMScheduler(num_train_timesteps=20, beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2', \
                                variance_type='fixed_small')
# Example usage
policy = DiffusionUnetLowdimPolicy2D(model, noise_scheduler, horizon=1, obs_dim=1, action_dim=2, n_action_steps=10, n_obs_steps=1)
data = policy.generate_data(num_samples=1000)
# policy.set_normalizer(LinearNormalizer().fit(data))

num_epochs = 100
batch_size = 32
lr = 1e-4

train_policy(policy, data, num_epochs, batch_size, lr)

obs = data['obs'][0]  # Take the first observation as an example
ani = policy.animate_sampling(obs)
ani.save('diffusion_sampling.gif', writer='pillow')
