import numpy as np
import pickle as pkl
import torch

patch_path = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/untar_pert_0.0625_epoch_10_mean_score_1.0_robot0_eye_in_hand_image.pkl'
# load the patch
patch = pkl.load(open(patch_path, 'rb'))
# random patch with the same shape as the patch with range of -0.0625 to 0.0625
random_patch = np.random.rand(*patch.shape) * 0.125 - 0.0625
random_patch[0] = 0
# save the random patch
# convert to torch tensor
random_patch = torch.tensor(random_patch, dtype=torch.float32)
pkl.dump(random_patch, open('random_patch.pkl', 'wb'))