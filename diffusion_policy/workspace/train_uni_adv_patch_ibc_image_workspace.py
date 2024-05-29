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
import dill
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.train_robomimic_image_workspace import TrainRobomimicImageWorkspace
from diffusion_policy.workspace.train_ibc_dfo_hybrid_workspace import TrainIbcDfoHybridWorkspace
from diffusion_policy.utils.attack_utils import transform_square_patch
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainUniAdvPatchIbcImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.checkpoint = cfg.checkpoint

        # configure model
        payload = torch.load(open(self.checkpoint, 'rb'), pickle_module=dill)
        cfg_loaded = payload['cfg']
        self.action_space = cfg_loaded.shape_meta.action.shape

        cls = hydra.utils.get_class(cfg_loaded._target_)
        workspace = cls(cfg_loaded, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.model = workspace.model
        # configure training state
        self.global_step = 0
        self.epoch = 0
        cfg.task.env_runner.n_envs = 1


    def run(self):

        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
    
        assert isinstance(dataset, BaseImageDataset)
        # set the dataloader num of workers to min of specified and available
        cfg.dataloader.num_workers = min(cfg.dataloader.num_workers, os.cpu_count())
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        self.model.eval()

        self.adv_patch = torch.zeros((1, 3, 84, 84)).to(self.model.device)
        # create an initial mask for the patch at top left corner
        mask = torch.zeros((84, 84)).to(self.model.device)
        mask[:cfg.patch_size, :cfg.patch_size] = 1

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)


        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device))
                        # transform patch
                        self.adv_patch, mask = transform_square_patch(patch = self.adv_patch, mask = mask, \
                                patch_shape = (cfg.patch_size, cfg.patch_size), \
                                data_shape = batch['obs']['robot0_eye_in_hand_image'].shape)
                        self.adv_patch, loss = self.model.train_adv_patch(batch, self.adv_patch, mask, cfg)
                        train_losses.append(loss)
                        tepoch.set_postfix(loss=loss)
                    print(f"Epoch {self.epoch} train loss: {np.mean(train_losses)}")
                    self.epoch += 1
                # ========= eval for this epoch ==========
                # policy = self.model
                # policy.eval()

                # run rollout
                # runner_log = env_runner.run(policy)
                # print(runner_log)

                # add the adversarial patch to the batch and evaluate
                # runner_log = env_runner.run(policy, adv_patch=self.adv_patch, cfg=cfg)
                # print(runner_log)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainUniAdvPatchIbcImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    main()
