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
from torch.utils.data import Dataset
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.utils.attack_utils import optimize_linear, clip_perturb
from diffusion_policy.utils.plot_utils import render_side_by_side_video
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import pickle

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

class ObservationActionDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        action = self.actions[idx]
        return observation, action

class AdvPatchRobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

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

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
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

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
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

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
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

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)
        self.single_env = env_fns[0]()

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def collect_data(self, policy: BaseImagePolicy, env, cfg=None):
        """
        Collect data by rolling out the policy for training adversarial patch
        """
        obs_list = []
        action_list = []
        done = False
        timestep = 0
        max_timesteps = cfg.max_steps
        obs = env.reset()
        with tqdm.tqdm(total=max_timesteps, desc="Collecting data", leave=False, mininterval=self.tqdm_interval_sec) as pbar:
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device=policy.device))
                obs_list.append(obs_dict)
                if cfg.algo == 'lstm_gmm':
                    action_dist = policy.action_dist(obs_dict)
                    action_means = action_dist.component_distribution.base_dist.loc
                    action_dict = policy.predict_action(obs_dict)
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                    # action means as a numpy array with dtype float32
                    action_means = action_means.squeeze(0).cpu().detach().numpy()
                    action_means = np.array(action_means, dtype=np.float32)
                    action_list.append(action_means)
                    # action = np.expand_dims(action, axis=0)
                    obs, reward, done, _ = env.step(action)
                    timestep += 1
                    pbar.update(1)
                elif cfg.algo == 'bc':
                    action_dict = policy.predict_action(obs_dict)
                    action = action_dict['action']
                    action_list.append(action.squeeze(0).cpu().detach().numpy())
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action']
                    obs, reward, done, _ = env.step(action)
                    timestep += 1
                    pbar.update(1)
                elif cfg.algo == 'ibc':
                    raise NotImplementedError("IBC not completed")
                else:
                    raise ValueError("Invalid algorithm")
                if timestep >= max_timesteps:
                    break
        pbar.close()
        # squeeze the shape of the obs_list
        obs_list = [dict_apply(obs, lambda x: x.squeeze(0)) for obs in obs_list]
        target_action_list = []
        for i in range(len(action_list)):
            target_action = action_list[i] + np.array(cfg.perturbations, dtype=np.float32)
            target_action_list.append(target_action)
        return obs_list, target_action_list
    
    def place_patch(self, image, patch, location=(0, 0)):
        # image: (B, n_obs, 3, size, size)
        # patch: (3, patch_size, patch_size)
        # location: (x, y)
        original_len = len(image.shape)
        if original_len == 4:
            image = image.unsqueeze(0)
        x, y = location
        image[:, :, :, x:x+patch.shape[1], y:y+patch.shape[2]] = self.patch_forward(patch)
        if original_len == 4:
            image = image.squeeze(0)
        return image

    def patch_forward(self, patch):
        # applies tanh to the patch and scales it to [0, 1]
        patch = torch.tanh(patch) / 2 + 0.5
        return patch


    def train_patch(self, policy: BaseImagePolicy, obs_list, action_list, cfg=None):
        """
        Takes in the policy and the observations and actions and trains the adversarial patch
        """
        dataset = ObservationActionDataset(obs_list, action_list)
        train_data, val_data = torch.utils.data.random_split(dataset, [int(0.85*len(dataset)), len(dataset)-int(0.85*len(dataset))])
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
        patch = torch.zeros((3, cfg.patch_size, cfg.patch_size),device=policy.device)
        patch_2 = torch.zeros((3, cfg.patch_size, cfg.patch_size),device=policy.device)
        patch = torch.nn.Parameter(patch)
        patch_2 = torch.nn.Parameter(patch_2)
        optimizer = torch.optim.SGD([patch, patch_2], lr=cfg.epsilon)
        loss = torch.nn.MSELoss()
        losses = []
        with tqdm.tqdm(total=cfg.n_epochs, desc="Training adversarial patch", leave=False, mininterval=self.tqdm_interval_sec) as pbar:
            for epoch in range(cfg.n_epochs):
                total_loss = 0
                for obs, target_action in dataloader:
                    policy.reset()
                    clean_obs = obs.copy()
                    obs[cfg.view] = self.place_patch(obs[cfg.view], patch, location=(cfg.x_loc, cfg.y_loc))
                    obs['agentview_image'] = self.place_patch(obs['agentview_image'], patch_2, location=(cfg.x_loc, cfg.y_loc))
                    if cfg.algo == 'lstm_gmm':
                        action_dist = policy.action_dist(obs)
                        action_means = action_dist.component_distribution.base_dist.loc.to(policy.device)
                        with torch.no_grad():
                            target_dist = policy.action_dist(clean_obs)
                            target_means = target_dist.component_distribution.base_dist.loc.to(policy.device)
                            target_action = target_means + torch.tensor(cfg.perturbations, device=policy.device, dtype=policy.dtype)
                        # target_action = target_action.to(policy.device)
                        loss_train = loss(action_means, target_action)
                    else:
                        raise NotImplementedError("BC not completed")
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    total_loss += loss_train.item()
                print(f"total loss: {total_loss}")
                losses.append(total_loss)
                # check validation loss
                val_loss = 0
                for obs, target_action in val_dataloader:
                    policy.reset()
                    obs[cfg.view] = self.place_patch(obs[cfg.view], patch, location=(cfg.x_loc, cfg.y_loc))
                    if cfg.algo == 'lstm_gmm':
                        action_dist = policy.action_dist(obs)
                        action_means = action_dist.component_distribution.base_dist.loc.to(policy.device)
                        target_action = action_means + torch.tensor(cfg.perturbations, device=policy.device, dtype=policy.dtype)
                        # target_action = target_action.to(policy.device)
                        loss_val = loss(action_means, target_action)
                    else:
                        raise NotImplementedError("BC not completed")
                    val_loss += loss_val.item()
                # save the patch for each epoch to select the best patch
                # with open(f"{cfg.patch_file[:-4]}_{epoch}.pkl", 'wb') as f:
                #     pickle.dump(patch, f)
                if cfg.log:
                    wandb.log({"Loss_train_patch": total_loss})
                    wandb.log({"Loss_val_patch": val_loss})
                pbar.update(1)
        pbar.close()
        return patch


    def patch_forward(self, patch, cfg):
        # applies tanh to the patch and scales it to [0, 1]
        # patch = torch.tanh(patch) / 2 + 0.5
        patch = cfg.epsilon * torch.tanh(patch)
        return patch

    def place_transparent_patch(self, image, patch, location=(0, 0)):
        # image: (B, n_obs, 3, size, size)
        # patch: (3, patch_size, patch_size)
        # location: (x, y)
        original_len = len(image.shape)
        if original_len == 4:
            image = image.unsqueeze(0)
        x, y = location
        # patch = patch_forward(patch)
        # image[:, :, :, x:x+patch.shape[1], y:y+patch.shape[2]] = patch_alpha * patch[:3] + (1 - patch_alpha) * image[:, :, :, x:x+patch.shape[1], y:y+patch.shape[2]]
        image[:, :, :, :, :] = image[:, :, :, :, :] + patch[:3]
        if original_len == 4:
            image = image.squeeze(0)
        return image


    def run(self, policy: BaseImagePolicy, adversarial_patch=None, cfg=None):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # check if data is already collected

        # train adversarial patch
        if cfg.patch_type == 'trained':
            # check if the patch is already trained
            if os.path.exists(cfg.patch_file):
                with open(cfg.patch_file, 'rb') as f:
                    patch = pickle.load(f)
                print(f"Patch loaded from {cfg.patch_file}")
            else:
                if os.path.exists(cfg.data_file):
                    with open(cfg.data_file, 'rb') as f:
                        obs_list, action_list = pickle.load(f)
                    print(f"Data loaded from {cfg.data_file} for {len(obs_list)} observations and {len(action_list)} actions")
                else:
                    # collect data
                    obs_list = []
                    action_list = []
                    for i in range(cfg.n_inits):
                        env_collect_data = self.env_fns[i]()
                        obs_list_iter, action_list_iter = self.collect_data(policy, env_collect_data, cfg)
                        # append the data to the list
                        obs_list.extend(obs_list_iter)
                        action_list.extend(action_list_iter)
                    # store the data as a file for future use
                    with open(cfg.data_file, 'wb') as f:
                        pickle.dump((obs_list, action_list), f)
                    print(f"Data collected for {len(obs_list)} observations and {len(action_list)} actions")
                patch = self.train_patch(policy, obs_list, action_list, cfg)
                # save the patch
                with open(cfg.patch_file, 'wb') as f:
                    pickle.dump(patch, f)
        elif cfg.patch_type == 'random':
            patch = torch.rand((3, cfg.patch_size, cfg.patch_size), device=device, dtype=dtype)
        elif cfg.patch_type == 'zero':
            patch = torch.zeros((3, cfg.patch_size, cfg.patch_size), device=device, dtype=dtype)
        elif cfg.patch_type == 'universal_transparent':
            # load the patch from the patch_file numpy file
            patch = np.load(cfg.patch_file, allow_pickle=True)
            # convert the patch to torch tensor and transfer to device
            if isinstance(patch, np.ndarray):
                patch = torch.from_numpy(patch).to(device=device)
                patch = self.patch_forward(patch, cfg)
            else:
                pass
        else:
            raise ValueError("Invalid patch type")
        
        observations = []


        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                # apply patch to the obs[cfg.view]
                if cfg.patch_type == 'universal_transparent':
                    assert cfg.x_loc == 0 and cfg.y_loc == 0
                    obs_dict[cfg.view] = self.place_transparent_patch(obs_dict[cfg.view], patch, location=(cfg.x_loc, cfg.y_loc))
                    # clamp the observation to [0, 1]
                    obs_dict[cfg.view] = torch.clamp(obs_dict[cfg.view], 0, 1)
                else:
                    obs_dict[cfg.view] = self.place_patch(obs_dict[cfg.view], patch, location=(cfg.x_loc, cfg.y_loc))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                observations.append(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # render side by side video
        # render_side_by_side_video(observations, os.path.join(self.output_dir, 'adversarial_universal_transparent_patch_video.mp4'), num_samples=5, cfg=cfg)
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
