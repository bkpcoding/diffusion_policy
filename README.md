# Adversarail Attacks on Behavior Cloning Policies

The project aims to develop Adversarial Attacks on Behavior Cloning Policies to check the adversarial robustness of these algorithms.
- Vanilla Behavior Cloning
- LSTM-GMM
- Implicit Behavior Cloning
- Diffusion Policy
- Vector Quantized - Behavior Transformer

The training dataset should go in `./data/robomimic/datasets/` and the checkpoints are stored in `./data/experiments/image/`. 

## Offline Attacks (Universal Adversarial Perturbations)
To run the `offline` attacks (Universal Adversarial Perturbations), you should first train the perturbation on the training dataset.
You should choose one of the checkpoints to train the perturbation on (I usually use `train_0`) from the config file, stored in `./diffusion_policy/config/`.
For all the `offline` attacks we are using the naming convention `train_univ_pert_{algo}_{state}_workspace.yaml`.
In here you have to take care to these parameters,
- `task` - The name of the task (you can find them under `config/tasks`)
- `_target_` - The name of target class for running the attack
- `epsilon` - The budget of your perturbation in Linf norm. (Note you need to double this for diffusion policy because I use they use the range from -1 to 1 for visualization, so after renormalizing the perturbation it becomes half in the observation space that we are visualizing)
- `epsilon_step` or `eps_iter` - The learning for your perturbation
- `view` - The name of the camera view that you want to attack. For example "robot0_eye_in_hand_image" or "agentview_image" or "both" for robomimic tasks. Or "image" for pusht tasks.
- `clip_min`, `clip_max` - The minimum and maximum range of the observation. This is usually 0 and 1, since we use this range after normalization. 
- `targeted` - Set to `True` if you want to run targeted attack, else `False`.
- `perturbations` - You need to set this when you are running targeted attack. This basically specifies the perturbation that you add to the actions to get your targeted action for running the targeted attack.
- `'log` - Whether to log or not. I use wandb for logging, you can change the name of the loggin project from the class specified in `_target_`. 

An example command to develop the adversarial pertubation for Diffusion Policy in offline mode.
```console
python train.py --config-name=train_univ_pert_diffusion_unet_hybrid_workspace.yaml
```

The perturbation is stored in the checkpoint folder of the policy used, the name is includes whether of the attack is targeted or untargeted, epsilon value
for the attack, epoch number at which it was saved, the mean test score for evaluation with the perturbation at that epoch, and the view of the image
that was attacked. 

After getting the perturbation you can evaluate the patch on the environments using the `eval_using_config.py` file, where you need to specify the config file for evaluation. 
The config file for this can be found in `./diffusion_policy/eval_configs/`, where you need to change the target to `RobomimicImageRunner` incase of robomimic experiments, put the name of the task, n_test, attack = True, **attack_type = 'patch'** and specify the absolute path of the perturbation that you trained during the above step in the patch_path. (You need to specify the project name for logging when running this file in `eval_using_config.py`.)


## Online Attacks
The `online` attacks develop the adversarial perturbation on a per timestep basis i.e., we develop a perturbation for that timestep and then work apply this perturbation for that timestep during evaluation and then get another perturbation for the next timestep, so in an online fashion.

The config for these experiments are stored in `./diffusion_policy/eval_configs`. The naming convention is `{algo}_{state}_{experiment_name}_{attack_type}_adversarial.yaml`. 
In this case, I have created a seperate name for the experiments and you can just copy most of this config and change the following parameters based on your requirements.

- `checkpoint`: Change this checkpoint to each seed and evaluate seperately as we are reporting both mean and std across seeds.
- `task`: The name of the task that you are testing for.
- `dataset_path`: The absoute path of the dataset for the task that you have chosen. 
- `n_envs`: The number of environments that you want to run in parallel at a time. Each environment takes around 1GB of memory.
- `n_test`: This is the number of environments that we are testing on, this should be 50. 
- `attack`: This should be True if you evaluating the policy with the attack, or False if you just want to run the policy without attack.
- `attack_type`: Set this to the attack type that you want to run, currently supports `fgsm` and `pgd`.
- `epsilons`: Choose from one of the epislons mentioned in the config, expect for DP currently I am using 0.0625 and for the DP 0.125. 
- `targeted`: True if you want to run targeted attack and False for untargeted attack. 
- `perturbations`: You need to set this when you are running targeted attack. This basically specifies the perturbation that you add to the actions to get your targeted action for running the targeted attack.
- `num_iter`: This is the number of inner iterations that you want to run for optimizing the perturbations for PGD, this is usually 40 for most of the algorithms except for DP in which case it is 4 (because we are applying them for multiple timesteps during denoising). 
- `attack_after_timesteps`: This is only need for DP, which specifies after how many timesteps we need to start attacking, since diffusion models learn a mean of the distribution during the initial timesteps and learn fine-grained structure afterwards. 
- `log`: Whether to log or not. You need to specify the name of the project that you want to log in in the `eval_using_config.py` file. 
- `save_video`: Whether to save the video or not, if True the video will be uploaded to wandb
- `n_vis`: Number of randomly sampled environment videos to be uploaded to wandb.


After specifying these parameters, run the command,
```console
python eval_using_config.py
```

