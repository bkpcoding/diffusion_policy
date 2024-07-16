import torch
import numpy as np
import matplotlib.pyplot as plt
from baukit import Trace
import pickle
import os
import pathlib
import hydra
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from baukit import show
from omegaconf import OmegaConf

train0 = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/lstm_gmm/train_0/checkpoints/epoch=1100-test_mean_score=1.000.ckpt'
train1 = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/lstm_gmm/train_1/checkpoints/epoch=0300-test_mean_score=1.000.ckpt'
train2 = '/teamspace/studios/this_studio/bc_attacks/diffusion_policy/data/experiments/image/lift_ph/lstm_gmm/train_2/checkpoints/epoch=0300-test_mean_score=1.000.ckpt'
device = 'cuda:0'

def return_img_encoder(checkpoint):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg_loaded = payload['cfg']
    cls = hydra.utils.get_class(cfg_loaded._target_)
    worskspace = cls(cfg_loaded, output_dir=None)
    worskspace.load_payload(payload, exclude_keys=None, include_keys=None)
    try:
        policy = worskspace.model
    except AttributeError:
        policy = worskspace.policy
    policy.to(device)
    policy.eval()
    try:
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    except:
        obs_encoder = policy.obs_encoder
    image_encoder = obs_encoder.obs_nets['robot0_eye_in_hand_image']
    return image_encoder

image_encoder1 = return_img_encoder(train0)
image_encoder2 = return_img_encoder(train1)
image_encoder3 = return_img_encoder(train2)

# check the weights of the encoder whether they are the same
def check_weights(image_encoder1, image_encoder2, image_encoder3):
    for key in image_encoder1.state_dict().keys():
        if not torch.equal(image_encoder1.state_dict()[key], image_encoder2.state_dict()[key]):
            print(f"Encoder1 and Encoder2 are different in {key}")
        if not torch.equal(image_encoder1.state_dict()[key], image_encoder3.state_dict()[key]):
            print(f"Encoder1 and Encoder3 are different in {key}")
        if not torch.equal(image_encoder2.state_dict()[key], image_encoder3.state_dict()[key]):
            print(f"Encoder2 and Encoder3 are different in {key}")

check_weights(image_encoder1, image_encoder2, image_encoder3)