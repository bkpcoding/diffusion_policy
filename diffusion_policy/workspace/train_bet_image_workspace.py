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
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.bet_image_policy import BETImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainBETImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.policy: BETImagePolicy
        self.policy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = self.policy.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        print(cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        print("Loading dataset")
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = None
        if cfg.training.enable_normalizer:
            normalizer = dataset.get_normalizer()
        else:
            normalizer = LinearNormalizer()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()


        self.policy.set_normalizer(normalizer)

        # fit action_ae (K-means)
        self.policy.fit_action_ae(
            normalizer['action'].normalize(dataset.get_all_actions())
        )
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
        print("Configuring environment runner")
        # configure env
        # cfg.task.env_runner.n_envs = 2
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
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
        self.policy.to(device)
        optimizer_to(self.optimizer, device)

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
        print(f"Training for {cfg.training.num_epochs} epochs")

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
                        raw_loss, loss_components = self.policy.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
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
                policy = self.policy
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch > 0:
                    runner_log = env_runner.run(policy)
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
                                loss, _ = self.policy.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and self.epoch > 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        n_samples = cfg.training.sample_max_batch
                        batch = dict_apply(train_sampling_batch, 
                            lambda x: x.to(device, non_blocking=True))
                        obs_dict = dict_apply(batch['obs'], lambda x: x[:n_samples])
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action']
                        start = cfg.n_obs_steps - 1
                        end = start + cfg.n_action_steps
                        gt_action = gt_action[:,start:end]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
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
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
import dill
import pickle 

class TrainBETUniPertImageWorkspaceDP(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        checkpoint = cfg.checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg_loaded = payload['cfg']

        cls = hydra.utils.get_class(cfg_loaded._target_)
        workspace = cls(cfg_loaded, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        try:
            self.model = workspace.model
        except:
            self.model = workspace.policy
        self.model.to(torch.device(cfg.training.device))

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        view = cfg.view
        device = cfg.training.device

        print("Loading dataset")
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = None
        if cfg.training.enable_normalizer:
            normalizer = dataset.get_normalizer()
        else:
            normalizer = LinearNormalizer()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()


        self.model.set_normalizer(normalizer)

        # fit action_ae (K-means)
        self.model.fit_action_ae(
            normalizer['action'].normalize(dataset.get_all_actions())
        )
        print("Configuring environment runner")
        # configure env
        # cfg.task.env_runner.n_envs = 2
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        if cfg.log:
            wandb.init(
                project="offline_bc_evaluation",
                name=f"BET_{cfg.epsilon}_targeted_{cfg.targeted}_view_{view}"
            )
            wandb.log({"epsilon": cfg.epsilon, "epsilon_step": cfg.epsilon_step, "targeted": cfg.targeted, "view": view})        # configure checkpoint
        self.model.eval()

        # device transfer
        self.model.to(device)

        # define the perturbation
        self.univ_pert = torch.zeros((3, 84, 84)).to(device)
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
        print(f"Training for {cfg.training.num_epochs} epochs")

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                loss_per_epoch = 0
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        self.model.zero_grad()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        obs = batch['obs']
                        if batch['obs']['agentview_image'].shape[0] != cfg.dataloader.batch_size:
                            continue
                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        obs[view] = obs[view] + self.univ_pert
                        # clamp the observation to be between clip_min and clip_max
                        obs[view] = torch.clamp(obs[view], cfg.clip_min, cfg.clip_max)
                        obs[view].requires_grad = True
                        if cfg.targeted:
                            batch['obs'] = obs
                            batch['action'] = batch['action'].to(device) + torch.tensor(cfg.perturbations, device =device)
                            raw_loss, loss_components = self.model.compute_loss(batch)
                            raw_loss = -raw_loss
                        else:
                            batch['obs'] = obs
                            raw_loss, loss_components = self.model.compute_loss(batch)
                        # compute loss
                        loss = raw_loss
                        if self.epoch == 0:
                            loss_per_epoch += loss.item()
                            continue
                        loss.backward()
                        loss_per_epoch += loss.item()
                        # update the perturbation
                        self.univ_pert = self.univ_pert + cfg.epsilon_step * torch.sum(obs[view].grad.sign(), dim=0)
                        # clip the perturbation
                        self.univ_pert = torch.clamp(self.univ_pert, -cfg.epsilon, cfg.epsilon)
                print(f"Loss per epoch: {loss_per_epoch}")
                if cfg.log:
                    wandb.log({"loss_per_epoch": loss_per_epoch, "epoch": self.epoch})
                print(f"L2 norm of the perturbation: {torch.norm(self.univ_pert)}")

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch > 0:
                    runner_log = env_runner.run(self.model, self.univ_pert, cfg)
                    # log all
                    step_log.update(runner_log)
                    test_mean_score= runner_log['test/mean_score']
                    print(f"Test mean score: {test_mean_score}")
                    if cfg.log:
                        wandb.log({"test_mean_score": test_mean_score, "epoch": self.epoch})
                    if cfg.targeted:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'tar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}.pkl')
                    else:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'untar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}.pkl')
                    pickle.dump(self.univ_pert, open(patch_path, 'wb'))
                self.epoch += 1
                json_logger.log(step_log)
                self.global_step += 1
        wandb.finish()                



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainIbcDfoHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

