if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import torch.utils
from torch.utils.data import DataLoader
import copy
import random
import torch.utils.data
import wandb
import tqdm
import numpy as np
import shutil
import pickle
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.policy.vq_bet_image_policy import VQBeTPolicy
from diffusion_policy.model.common.lr_scheduler import get_scheduler
OmegaConf.register_new_resolver("eval", eval, replace=True)
import os
os.environ["NUMEXPR_MAX_THREADS"] = "12"
import numexpr

class TrainVQBeTImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: VQBeTPolicy = hydra.utils.instantiate(cfg.policy)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

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
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )


        # configure logging
        if cfg.log:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

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
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        loss_dict = self.model.compute_loss(batch, epoch=local_epoch_idx)
                        loss = loss_dict['loss']
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # logging 
                        loss_cpu = loss.item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        if self.global_step > cfg.policy.config.n_vqvae_training_steps:
                            step_log = {
                                'train_loss': loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': self.optimizer.param_groups[0]['lr'],
                                'classification_loss': loss_dict['classification_loss'],
                                'offset_loss': loss_dict['offset_loss'],
                                'vq_action_error': loss_dict['vq_action_error'],
                                'offset_action_error': loss_dict['offset_action_error'],
                                'action_error_max': loss_dict['action_error_max'],
                                'action_mse_error': loss_dict['action_mse_error'],
                            }
                        else:
                            step_log = {
                                'train_loss': loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': self.optimizer.param_groups[0]['lr'],
                            }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if cfg.log:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                self.model.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch > 0:
                    runner_log = env_runner.run(self.model, cfg=cfg)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and self.epoch > 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                info = self.model.compute_loss(batch, epoch=self.epoch)
                                loss = info['loss']
                                classification_loss = info['classification_loss']
                                offset_loss = info['offset_loss']
                                vq_action_error = info['vq_action_error']
                                offset_action_error = info['offset_action_error']
                                action_error_max = info['action_error_max']
                                action_mse_error = info['action_mse_error']
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses))
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            step_log['val_classification_loss'] = classification_loss
                            step_log['val_offset_loss'] = offset_loss
                            step_log['val_vq_action_error'] = vq_action_error
                            step_log['val_offset_action_error'] = offset_action_error
                            step_log['val_action_error_max'] = action_error_max
                            step_log['val_action_mse_error'] = action_mse_error

                if (self.epoch % cfg.training.sample_every) == 0 and self.epoch > 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        T = gt_action.shape[1]

                        pred_actions = list()
                        self.model.reset()
                        for i in range(T):
                            result = self.model.predict_action(
                                dict_apply(obs_dict, lambda x: x[:,[i]])
                            )
                            pred_actions.append(result['action'])
                        pred_actions = torch.cat(pred_actions, dim=1)
                        mse = torch.nn.functional.mse_loss(pred_actions, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_actions
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and self.epoch > 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if cfg.log:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
    wandb.finish()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRobomimicImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()