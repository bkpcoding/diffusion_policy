import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
import os

def render_side_by_side_video(observations, output_path, num_samples=5, cfg=None):
    # Sample random indices
    batch_size = observations[0]['agentview_image'].shape[0]
    random_indices = random.sample(range(batch_size), num_samples)
    
    # Extract frames and combine them
    combined_frames = []
    for idx in random_indices:
        for obs in observations:
            agentview_frames = obs['agentview_image'][idx]  # shape: [obs_horizon, C, H, W]
            eye_in_hand_frames = obs['robot0_eye_in_hand_image'][idx]  # shape: [obs_horizon, C, H, W]
            # sample every other frame
            agentview_frames = agentview_frames[::2]
            eye_in_hand_frames = eye_in_hand_frames[::2]
            for agentview_frame, eye_in_hand_frame in zip(agentview_frames, eye_in_hand_frames):
                agentview_frame = agentview_frame.detach().cpu().numpy()
                eye_in_hand_frame = eye_in_hand_frame.detach().cpu().numpy()
                agentview_frame = np.transpose(agentview_frame, (1, 2, 0))  # Convert to HWC format
                eye_in_hand_frame = np.transpose(eye_in_hand_frame, (1, 2, 0))  # Convert to HWC format
                
                # Combine frames side by side
                combined_frame = np.concatenate((agentview_frame, eye_in_hand_frame), axis=1)
                combined_frames.append(combined_frame)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    ims = []
    
    for frame in combined_frames:
        im = ax.imshow(frame, animated=True)
        ims.append([im])
    
    # Create an animation
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    
    # Save the animation as a video
    ani.save(output_path, writer='ffmpeg')
    
    # Save video to wandb
    if cfg.log:
        wandb.log({"video": wandb.Video(output_path)})

# Example usage
if __name__ == '__main__':
    observations = [
        {
            'agentview_image': np.random.rand(10, 5, 3, 64, 64),  # Example shape [batch_size, obs_horizon, C, H, W]
            'robot0_eye_in_hand_image': np.random.rand(10, 5, 3, 64, 64)
        },
        # Add more observation dicts if needed
    ]

    output_path = os.path.join('/teamspace/studios/this_studio/bc_attacks/diffusion_policy', 'test.mp4')
    render_side_by_side_video(observations, output_path, num_samples=5, cfg=None)