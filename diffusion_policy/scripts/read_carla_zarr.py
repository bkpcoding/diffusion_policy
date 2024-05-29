if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import click
import numpy as np
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr

@click.command()
@click.option('--input', '-i', required=True)
@click.option('--dt', default=0.1, type=float)
def main(input, dt):
    z1 = zarr.open(input, mode='r')
    print(z1.tree())

if __name__ == '__main__':
    main()
