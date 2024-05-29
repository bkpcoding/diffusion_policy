if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import click
import pathlib
import numpy as np
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.option('-i', '--input', required=True, help='input dir contains npy files')
@click.option('-o', '--output', required=True, help='output zarr path')
def main(input, output):
    data_directory = pathlib.Path(input)
    observations = torch.load(
        data_directory / "all_observations.pth"
    )
    actions = torch.load(data_directory / "all_actions_pm1.pth")
    eps_length = torch.load(data_directory / "seq_lengths.pth")

    buffer = ReplayBuffer.create_empty_numpy()
    for i in range(len(eps_length)):
        eps_len = int(eps_length[i])
        # convert the tensor to numpy with the correct dtype
        obs = observations[i,:eps_len].clone().detach().numpy().astype(np.float32)
        action = actions[i,:eps_len].clone().detach().numpy().astype(np.float32)
        print(f"Observation and action shape: {obs.shape}, {action.shape}")
        data = {
            'obs': obs,
            'action': action
        }
        buffer.add_episode(data)

    print(f"Saving to {output} this might take some time ...")
    buffer.save_to_path(zarr_path=output, chunk_length=-1)

if __name__ == '__main__':
    main()
