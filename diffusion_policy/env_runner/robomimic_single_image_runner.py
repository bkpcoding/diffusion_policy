import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.utils.attack_utils import optimize_linear, clip_perturb

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    print(env_meta)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicSingleImageRunner(BaseImageRunner):
    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        def init_fn(env, seed=42,
            enable_render=True):
            # setup rendering
            # video_wrapper
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if enable_render:
                filename = pathlib.Path(output_dir).joinpath(
                    'media', wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env.env.file_path = filename

            # switch to seed reset
            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None
            env.seed(seed)

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
            return MultiStepWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        self.env = env_fn()
    
    def create_videos(self, policy:BaseImagePolicy, cfg=None, perturbation=None):
        self.env.seed(cfg.seed)
        # init fn
        self.init_fn = lambda env, seed: init_fn(env, seed, enable_render=True)
        # self.env.set_mode('test')
        obs = self.env.reset()
        done = False
        policy.reset()
        timestep = 0
        observations = []
        while not done:
            print(f'timestep: {timestep}')
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(policy.device))
            obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0))
            observations.append(obs_dict['agentview_image'].squeeze(0).detach().cpu().numpy())
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action'].squeeze(0)
            if perturbation is not None:
                action_perturbation = np.array([-1*perturbation, perturbation, 0, 0, 0, 0, 0]).reshape(action.shape)
                action = action + action_perturbation
            obs, reward, done, info = self.env.step(action)
            timestep += 1
            if timestep % 100 == 0:
                print(f'timestep: {timestep}')
                break
        self.env.close()
        # create a video from the observations
        observations = np.array(observations)
        ims = []
        fig, ax = plt.subplots()
        ax.axis('off')
        for i in range(len(observations)):
            im = ax.imshow(observations[i][0, :, :, :].transpose(1, 2, 0))
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save(f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/videos/perturbed_pick_0.1_two_perturb.gif', writer='imagemagick', fps=4)
        


    def run(self, policy:BaseImagePolicy, epsilon=0.1, cfg=None):
        # assert mode in ['train', 'test']
        self.env.seed(cfg.seed)
        # init fn
        self.init_fn = lambda env, seed: init_fn(env, seed, enable_render=False)
        # self.env.set_mode('test')

        views = ['agentview_image', 'robot0_eye_in_hand_image']
        for view in views:
            self.env.reset()
            obs = self.env.reset()
            done = False
            policy.reset()
            epsilon = 0.1
            timestep = 0
            clean_energy_list = []
            noisy_energy_list = []
            while not done:
                tqdm.tqdm.write(f'timestep: {timestep}')
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(policy.device))
                # increase the dimension at the begining of the keys in obs_dict
                obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, return_energy=True)
                    noisy_obs_dict = dict_apply(obs_dict, lambda x: x + epsilon * torch.randn_like(x) if x == view else x)
                    noisy_action_dict = policy.predict_action(noisy_obs_dict, return_energy=True)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                np_noisy_action_dict = dict_apply(noisy_action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # print(action)
                energy = np_action_dict['energy'].squeeze(0)
                samples = np_action_dict['samples'].squeeze(0)
                idx_action = energy.argmax()
                sampled_action = samples[idx_action]
                obs, reward, done, info = self.env.step(action)

                noisy_action = np_noisy_action_dict['action'].squeeze(0)
                noisy_energy = np_noisy_action_dict['energy'].squeeze(0)
                noisy_samples = np_noisy_action_dict['samples'].squeeze(0)
                noisy_idx_action = noisy_energy.argmax()
                noisy_sampled_action = noisy_samples[noisy_idx_action]
                print(f'energy: {energy[idx_action]}, noisy_energy: {noisy_energy[noisy_idx_action]}')
                print(f'action: {action}, noisy_action: {noisy_action}')
                # print(f'sampled_action: {sampled_action}, noisy_sampled_action: {noisy_sampled_action}')
                clean_energy_list.append(energy[idx_action])
                noisy_energy_list.append(noisy_energy[noisy_idx_action])
                timestep += 1
        
            # plot the energy for clean and noisy over time
            fig = plt.figure()
            plt.plot(clean_energy_list, label='clean')
            plt.plot(noisy_energy_list, label='noisy')
            plt.legend()
            plt.savefig(f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/images/random_noise{epsilon}_{view}.png')
            plt.close(fig)


    def apply_fgsm_attack_loss_of_ibc(self, obs_dict_cp, policy:BaseImagePolicy, cfg, actions = None, \
                                        action_samples=None, patch=None, mask=None):
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        obs_dict_copy = dict_apply(obs_dict_cp, lambda x: x.clone().detach().requires_grad_(True))
        if cfg.attack_type == 'patch':
            if cfg.patch_location == 'top_left':
                    obs_dict_copy['robot0_eye_in_hand_image'] = mask * patch + (1 - mask) * obs_dict_cp['robot0_eye_in_hand_image']
            else:
                raise ValueError("Patch location not supported")

        # create a copy of the original obs_dict that requires grad for backward pass                    
        # obs_dict_copy = dict_apply(obs_dict_cp, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()

        assert actions is not None, "Actions must be provided for IBC loss"
        if action_samples is not None:
            loss, obs_dict_copy, obs_features = policy.compute_loss_with_grad(obs_dict_copy, actions, action_samples)
        else:
            loss, obs_dict_copy, obs_features = policy.compute_loss_with_grad(obs_dict_copy, actions)
        if cfg.targeted:
            loss = -loss
        self.loss_per_iteration.append(loss.item())
        loss.backward()
        # print("Loss: ", loss.item())
        # print("Observation features and its grad ", obs_features, obs_features.grad)
        prev_obs_dict = obs_dict_cp
        for view in views:
            # check if the grad is not None
            assert obs_dict_copy[view].grad is not None, "Input tensor does not have a grad"
            if cfg.attack_type == 'patch':
                if view == 'robot0_eye_in_hand_image':
                    grad = torch.sign(patch.grad)
                    # print("Grad before mask: ", grad)
                    grad = cfg.eps_iter * mask * grad
                    # print("GRad after mask: ", grad)    
                    patch = patch + grad
                    obs_dict_copy[view] = (1 - mask) * obs_dict_copy[view] + mask * patch
                    obs_dict_copy[view] = torch.clamp(obs_dict_copy[view], cfg.clip_min, cfg.clip_max)
                    continue
                else:
                    continue
            # assert statement to check that this part isn't reached when using patch attack
            assert not cfg.attack_type == 'patch', "Patch attack not supported for this part of the code"
            # calculate the average value of the grad and store in wandb
            grad = torch.sign(obs_dict_copy[view].grad)
            if obs_dict_copy[view].grad is not None:
                obs_dict_copy[view].grad.zero_()  # Clear gradients for obs_dict to save memory
            # log the lipschitz constant for the input tensor
            lip_grad = obs_dict_copy[view].grad.view(obs_dict_copy[view].shape[0], -1)
            obs_dict_copy[view] = obs_dict_copy[view] + optimize_linear(grad, cfg.eps_iter, cfg.norm)
            old_obs_dict = obs_dict_copy[view]
            if (cfg.clip_min is not None) or (cfg.clip_max is not None):
                if cfg.clip_min is None or cfg.clip_max is None:
                    raise ValueError(
                        "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                    )
                obs_dict_copy[view] = torch.clamp(obs_dict_copy[view], cfg.clip_min, cfg.clip_max)
        return obs_dict_copy


    def apply_pgd_attck(self, policy, obs_dict, cfg, epsilon, clean_action, action_samples):
        '''
        Apply the PGD attack on the observation
        '''
        self.loss_per_iteration = []
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")

        adv_obs_dict = obs_dict
        eps_iter = cfg.eps_iter
        num_iter = cfg.num_iter
        for i in range(num_iter):
            adv_obs_dict = self.apply_fgsm_attack_loss_of_ibc(adv_obs_dict, policy, cfg, clean_action, action_samples)
            for view in views:
                    perturbation = adv_obs_dict[view] - obs_dict[view]
                    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                    adv_obs_dict[view] = obs_dict[view] + perturbation
                    adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], cfg.clip_min, cfg.clip_max)
        return adv_obs_dict
        
    def get_patch(self, patch_size, patch_type, image_size):
        '''
        Get the patch based on the patch type
        '''
        if patch_type == 'square':
            patch = torch.zeros((3, image_size, image_size))
            mask = torch.zeros((3, image_size, image_size))
            mask[:, :patch_size, :patch_size] = 1
        else:
            raise ValueError("Patch type not supported")
        return patch, mask


    def apply_patch_attack(self, policy, obs_dict, cfg, epsilon, clean_action, action_samples):
        '''
        Apply the patch attack on the observation
        '''
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        self.loss_per_iteration = []
        # get the image size
        image_size = obs_dict['robot0_eye_in_hand_image'].shape[-1]
        # get the patch size
        patch_size = cfg.patch_size
        # get the patch location
        patch_location = cfg.patch_loc
        # get the patch type
        patch_type = cfg.patch_type
        patch, mask = self.get_patch(patch_size, patch_type, image_size)
        # make the patch requires grad
        patch = patch.requires_grad_()
        adv_obs_dict = obs_dict
        
        for i in range(cfg.num_iter):
            adv_obs_dict = self.apply_fgsm_attack_loss_of_ibc(adv_obs_dict, policy, cfg, clean_action, action_samples, patch, mask)
            # for view in views:
            #         perturbation = adv_obs_dict[view] - obs_dict[view]
            #         perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            #         adv_obs_dict[view] = obs_dict[view] + perturbation
            #         adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], cfg.clip_min, cfg.clip_max)
        return adv_obs_dict
        


    def probability_of_action(self, policy:BaseImagePolicy, cfg = None):
        '''
        We plot the probability of taking the same action under different perturbations
        '''
        self.env.seed(cfg.seed)
        # init fn
        self.init_fn = lambda env, seed: init_fn(env, seed, enable_render=False)
        obs = self.env.reset()
        done = False

        views = ['agentview_image', 'robot0_eye_in_hand_image']
        timestep = 0
        observations = []
        while not done:
            print(f'timestep: {timestep}')
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(policy.device))
            obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0))
            observations.append(obs_dict['agentview_image'].squeeze(0).detach().cpu().numpy())
            probs = {}
            distance = {}
            epsilon = cfg.epsilon
            policy.reset()
            B = obs_dict['agentview_image'].shape[0]
            T_neg = policy.train_n_neg
            T_a = policy.n_action_steps
            naction_stats = policy.get_naction_stats()
            action_samples = torch.distributions.Uniform(
                low=naction_stats['min'],
                high=naction_stats['max']
            ).sample((B, T_neg, T_a)).to(device=obs_dict['agentview_image'].device)
            loss_per_perturbation = {}
            for perturbation in cfg.perturbations:
                with torch.no_grad():
                    clean_action = policy.predict_action(obs_dict)['action']
                    clean_action_before = clean_action.clone()
                    if cfg.targeted:
                        # make the action all zeros except the second coordinate to -1
                        # making the robot move in a horizontal direction
                        clean_action = clean_action + torch.tensor([0, perturbation, 0, 0, 0, 0, 0]).to(clean_action.device)
                clean_action_np = clean_action.detach().cpu().numpy()
                action_samples = torch.cat([clean_action.unsqueeze(1), action_samples], dim=1)
                action_samples[:, 1] = clean_action_before
                if cfg.attack_type == 'pgd':
                    adv_obs = self.apply_pgd_attck(policy, obs_dict, cfg, epsilon, clean_action, action_samples)
                    for i in range(3):
                        new_action_samples = torch.distributions.Uniform(
                            low=naction_stats['min'],
                            high=naction_stats['max']
                        ).sample((B, T_neg, T_a)).to(device=obs_dict['agentview_image'].device)
                        new_action_samples = torch.cat([clean_action.unsqueeze(1), new_action_samples], dim=1)
                        new_action_samples[:, 1] = clean_action_before
                        adv_obs = self.apply_pgd_attck(policy, adv_obs, cfg, epsilon, clean_action, new_action_samples)
                elif cfg.attack_type == 'patch':
                    adv_obs = self.apply_patch_attack(policy, obs_dict, cfg, epsilon, clean_action, action_samples)
                    # save the perturbed image
                    perturbed_image = adv_obs['robot0_eye_in_hand_image'].squeeze(0).detach().cpu().numpy()[0]
                    print(max(perturbed_image.flatten()), min(perturbed_image.flatten()))
                    plt.imsave(f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/images/adversarial_patch/patch_attack_{perturbation}.png', perturbed_image.transpose(1, 2, 0))
                # loss_per_perturbation[perturbation] = self.loss_per_iteration
                # print(f'Loss for epsilon {epsilon}: {self.loss_per_iteration}')
                # get the probability of taking the same action by running the
                # policy on the adversarial observation 100 times
                # n_samples = 100
                # action_list = []
                # for i in range(n_samples):
                #     action = policy.predict_action(adv_obs)['action']
                #     action_list.append(action.detach().cpu().numpy())
                # action_list = np.array(action_list)
                # # get the probability of taking similar action within the bound of 0.1 of the clean action
                # # in linf norm
                # diff = np.abs(action_list.reshape(action_list.shape[0], -1) - clean_action_np.reshape(1, -1))
                # diff = np.sum(diff, axis=1)
                # distance[perturbation] = np.mean(diff)
                # # print the linf norm of the difference
                # diff = np.sum(diff == 0, axis=0) / n_samples
                # # print(f'Probability of taking the same action for epsilon {epsilon}: {diff}')
                # probs[perturbation] = diff
            # predicted_output = policy.predict_action(obs_dict, return_energy=True)
            # predicted_energy = predicted_output['energy']
            # predicted_samples = predicted_output['samples']
            # predicted_action = predicted_output['action']
            # # print the sample and energy which has maximum energy
            # predicted_index = predicted_energy.argmax()
            # print(f"Timestep: {timestep}")
            # print(f'Predicted energy: {predicted_energy[0, predicted_index]}')
            # print(f'Predicted sample: {predicted_samples[0, predicted_index]}')
            # predicted_action_np = predicted_action.detach().cpu().numpy()
            adversarial_output = policy.predict_action(adv_obs, return_energy=True)
            # adversarial_energy = adversarial_output['energy']
            # adversarial_samples = adversarial_output['samples']
            # # print the sample and energy which has maximum energy
            # adversarial_index = adversarial_energy.argmax()
            # print(f"Timestep: {timestep}")
            # print(f'Adversarial energy: {adversarial_energy[0, adversarial_index]}')
            # print(f'Adversarial sample: {adversarial_samples[0, adversarial_index]}')
            # adversarial_action_index = 0
            # print(f"Adversarial action index: {adversarial_action_index}")
            # print(f'Adversarial action energy: {adversarial_energy[0, adversarial_action_index]}')
            # print(f'Adversarial action sample: {adversarial_samples[0, adversarial_action_index]}')
            adversarial_action = adversarial_output['action']
            adversarial_action_np = adversarial_action.detach().cpu().numpy()
            # obs, reward, done, info = self.env.step(predicted_action_np.squeeze(0))
            obs, reward, done, info = self.env.step(adversarial_action_np.squeeze(0))
            timestep += 1
            # plot the probability of taking the same action vs epsilon
            # fig = plt.figure()
            # plt.plot(probs.keys(), probs.values())
            # # draw on the second y axis the distance
            # plt.twinx()
            # plt.plot(distance.keys(), distance.values(), 'r')
            # plt.xlabel('Perturbation')
            # plt.ylabel('Probability of taking the same action')
            # plt.savefig(f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/images/targeted_perturbations/probability_of_same_action_linf_0.0_act_inject_{timestep}.png')
            # plt.close(fig)

            # fig = plt.figure()
            # # plot the loss per epsilon
            # for perturbation, loss in loss_per_perturbation.items():
            #     plt.plot(loss, label=f'perturbation: {perturbation}')
            # plt.legend()
            # plt.xlabel('Iteration')
            # plt.ylabel('Loss')
            # plt.savefig(f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/images/targeted_perturbations/loss_per_epsilon_act_inject_{timestep}.png')
            # plt.close(fig)
            if timestep % 80 == 0:
                break
        self.env.close()
        # create a video from the observations
        observations = np.array(observations)
        ims = []
        fig, ax = plt.subplots()
        ax.axis('off')
        for i in range(len(observations)):
            im = ax.imshow(observations[i][0, :, :, :].transpose(1, 2, 0))
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save(f'/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/videos/target_perturbations_/multiple_samples/perturb_{cfg.perturbations[0]}_epsilon_{cfg.epsilon}_strength_{cfg.num_iter}.gif', writer='imagemagick', fps=4)

