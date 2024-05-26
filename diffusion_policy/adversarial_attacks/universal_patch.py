import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    # config is in one path up from this file ../config
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')
    )
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cfg.task.dataset_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath(cfg.task.dataset_path))
    dataset = RobomimicReplayImageDataset(shape_meta = cfg.task.dataset.shape_meta,
        dataset_path = cfg.task.dataset.dataset_path,
        horizon = cfg.task.dataset.horizon,
        pad_before = cfg.task.dataset.pad_before,
        pad_after=cfg.task.dataset.pad_after,
        n_obs_steps=cfg.task.dataset.n_obs_steps,
        abs_action=cfg.task.dataset.abs_action,
        # rotation_rep='rotation_6d', # ignored when abs_action=False
        rotation_rep=cfg.task.dataset.rotation_rep,
        use_legacy_normalizer=cfg.task.dataset.use_legacy_normalizer,
        use_cache=cfg.task.dataset.use_cache,
        seed=cfg.task.dataset.seed,
        val_ratio=cfg.task.dataset.val_ratio,
        )
    dataset = hydra.utils.call(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
 
    # configure policy


if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    main()
