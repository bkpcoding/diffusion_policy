from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import sys
import numpy as np
import pickle

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class BETImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            action_ae: nn.Module, 
            state_prior: nn.Module,
            horizon,
            n_action_steps,
            n_obs_steps,
            crop_shape=(76, 76),
            eval_fixed_crop=True,
            obs_encoder_group_norm=True,
            **kwargs):
        super().__init__()
    
        self.shape_meta = shape_meta
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        # self.obs_encoding_net = obs_encoding_net
        self.state_prior = state_prior
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc',
            hdf5_type='image',
            task_name='lift',
            dataset_type='ph',
            pretrained_backbone=True,)
        
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )
        print(f"Policy config: {config}")
        self.obs_encoder = obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        self.obs_feature_dim = obs_encoder.output_shape()[0]


    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], return_latent = False) -> Dict[str, torch.Tensor]:
        """
        Predict action given observation
        """
        obs = dict_apply(obs_dict, lambda x: x.to(self.device).requires_grad_(True))
        nobs = self.normalizer.normalize(obs)
        value = next(iter(nobs.values()))
        B, _, C, H, W = nobs['image'].shape
        B, To = value.shape[:2]
        T = self.horizon
        # To = self.n_obs_steps
        obs = torch.full((B,T,C,H,W), -2, dtype=nobs['image'].dtype, device=nobs['image'].device)
        agent_pos = torch.full((B, T, 2), -2, dtype=nobs['agent_pos'].dtype, device=nobs['agent_pos'].device)
        obs[:,:To,:] = nobs['image'][:,:To,:]
        agent_pos[:,:To,:] = nobs['agent_pos'][:,:To,:]
        obs = {'image': obs, 'agent_pos': agent_pos}
        # or 
        # nobs['image'][:,To:,:] = -2
        obs = dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(obs)
        # reshape back to B, T, Do
        nobs_features = nobs_features.reshape(-1, T, self.obs_feature_dim)
        latent = self.state_prior.generate_latents(nobs_features)
        output = self.state_prior.generate_latents(nobs_features, return_output=True)
        if return_latent:
            return {'latent': latent, 'output': output}
        action = self.action_ae.decode_actions(latent, nobs_features)
        # unnormalize the actions
        action = self.normalizer['action'].unnormalize(action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        # print("Start and end", start, end)
        action = action[:, start:end]

        return {'action': action, 'features': nobs_features}

    # ========= training  ============
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)

    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
                self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
            ) -> torch.optim.Optimizer:
            return self.state_prior.get_optimizer(
                    weight_decay=weight_decay, 
                    learning_rate=learning_rate, 
                    betas=tuple(betas))

    def compute_loss(self, batch, return_nobs_feat=False):
         # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        # mask out observations after n_obs_steps
        B = naction.shape[0]
        # To = self.n_obs_steps
        T = self.horizon
        this_nobs = dict_apply(nobs, 
            lambda x: x[:, :T, ...].reshape(-1, *x.shape[2:]))
        # nobs[:,To:,:] = -2 # (normal obs range [-1,1])
        # the shape of this_nobs['image'] now is 256*10 x 3 x 76 x 76 
        nobs_features = self.obs_encoder(this_nobs)
        # print(f'obs_features shape: {nobs_features.shape}')
        # nobs_features = self.obs_encoder(nobs)
        # reshape nobs_features to B, To, Do
        nobs_features = nobs_features.reshape(B, T, -1)
        # mask out observations after n_obs_steps
        nobs_features[:,self.n_obs_steps:,:] = -2
        # print(f'obs_features shape: {nobs_features.shape}')
        latent = self.action_ae.encode_into_latent(naction, nobs_features)
        # print(f'latent {latent}')
        _, loss, loss_components = self.state_prior.get_latent_and_loss(
            obs_rep=nobs_features,
            target_latents=latent,
            return_loss_components=True,
        )
        if return_nobs_feat:
            return loss, loss_components, nobs_features
        return loss, loss_components
