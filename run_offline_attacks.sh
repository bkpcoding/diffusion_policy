#!/bin/bash
cd /teamspace/studios/this_studio/bc_attacks/diffusion_policy
python train.py --config-name=train_univ_pert_bet_image_workspace.yaml training.seed=123 epsilon=0.0625 targeted=True

python train.py --config-name=train_univ_pert_bet_image_workspace.yaml training.seed=123 epsilon=0.0625 targeted=False
