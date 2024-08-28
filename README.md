# Adversarail Attacks on Behavior Cloning Policies

The project aims to develop Adversarial Attacks on Behavior Cloning Policies to check the adversarial robustness of these algorithms.
- Vanilla Behavior Cloning
- LSTM-GMM
- Implicit Behavior Cloning
- Diffusion Policy
- Vector Quantized - Behavior Transformer

The training dataset should go in `./data/robomimic/datasets/` and the checkpoints are stored in `./data/experiments/image/`. 

To run the `offline` attacks (Universal Adversarial Perturbations), you should first train the perturbation on the training dataset.
You should choose one of the checkpoints to train the perturbation on (I usually use `train_0`) from the config file, stored in `./diffusion_policy/config/`.
For all the `offline` attacks we are using the naming convention `train_univ_pert_{algo}_{state}_workspace.yaml`.
In here you have to take care to these parameters,
- `task` - The name of the task (you can find them under `config/tasks`)
- `target` - The name of target class for running the attack
- `epsilon` - The budget of your perturbation in Linf norm. (Note you need to double this for diffusion policy because I use they use the range from -1 to 1 for visualization, so after renormalizing the perturbation it becomes half in the observation space that we are visualizing)
- `epsilon_step` or `eps_iter` - The learning for your perturbation
- `view` - The name of the camera view that you want to attack. For example "robot0_eye_in_hand_image" or "agentview_image" or "both" for robomimic tasks. Or "image" for pusht tasks.
- `clip_min`, `clip_max` - The minimum and maximum range of the observation. This is usually 0 and 1, since we use this range after normalization. 
- `targeted` - Set to `True` if you want to run targeted attack, else `False`.
- `perturbations` - You need to set this when you are running targeted attack. This basically specifies the perturbation that you add to the actions to get your targeted action for running the targeted attack.
```console
python train.py --config-name=train_univ_pert_diffusion_unet_hybrid_workspace.yaml
```


The perturbation is stored in the checkpoint folder of the policy used, the name is includes whether of the attack is targeted or untargeted, epsilon value
for the attack, epoch number at which it was saved, the mean test score for evaluation with the perturbation at that epoch, and the view of the image
that was attacked. 



