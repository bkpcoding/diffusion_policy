import torch
import numpy as np


def optimize_linear(grad, eps, norm='linf'):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == 'linf':
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 'l1':
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 'l2':
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation



def clip_perturb(eta, norm, eps):
    """
    Clip the perturbation, so that the resulting perturbation is at most
    eps in the given norm.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in ['linf', 'l1', 'l2']:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == 'linf':
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 'l1':
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 'l2':
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta

def get_patch_positions(mask):
    # Find the coordinates where the mask is 1
    coords = torch.nonzero(mask == 1)
    
    # Get the top-left and bottom-right coordinates
    top_left = coords.min(dim=0).values
    bottom_right = coords.max(dim=0).values
    
    # Convert to tuples
    top_left = (top_left[0].item(), top_left[1].item())
    bottom_right = (bottom_right[0].item(), bottom_right[1].item())
    
    return top_left, bottom_right



def swap_mask_position(mask, patch_shape, top_left, bottom_right, new_top_left):
    """
    Swap the patch to a new position in the mask.
    
    Inputs:
    mask: torch.Tensor, shape (image_size, image_size). The mask tensor with 1s indicating the patch position.
    patch_shape: tuple. The shape of the patch. (patch_size, patch_size)
    new_top_left: tuple. The new top-left position for the patch (start_x, start_y).
    
    Returns:
    new_mask: torch.Tensor, shape (image_size, image_size). The updated mask tensor with the patch at the new position.
    """
    # Get the top-left and bottom-right positions of the original patch
     
    # Extract the patch from the mask
    patch = mask[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1].clone()
    
    # Set the original patch position in the mask to zero
    mask[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1] = 0
    
    # Calculate the new bottom-right position
    new_bottom_right = (new_top_left[0] + patch_shape[0] - 1, new_top_left[1] + patch_shape[1] - 1)
    
    # Place the extracted patch at the new position
    mask[new_top_left[0]:new_bottom_right[0] + 1, new_top_left[1]:new_bottom_right[1] + 1] = patch
    
    return mask


def swap_patch_position(patch, mask, patch_shape, top_left, bottom_right, new_top_left):
    """
    Swap the patch to a new position in the mask.

    Inputs:
    mask: torch.Tensor, shape (C, image_size, image_size). The mask tensor with 1s indicating the patch position.
    patch_shape: tuple. The shape of the patch. (C, patch_size, patch_size)
    new_top_left: tuple. The new top-left position for the patch (start_x, start_y).

    Returns:
    new_mask: torch.Tensor, shape (C, image_size, image_size). The updated mask tensor with the patch at the new position.
    """

    # Extract the patch from the mask
    patch_cp= patch[:, top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1].clone()

    # Set the original patch position in the mask to zero
    patch[:, top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1] = 0

    # Calculate the new bottom-right position
    new_bottom_right = (new_top_left[0] + patch_shape[0] - 1, new_top_left[1] + patch_shape[1] - 1)
    # print(f"New top left: {new_top_left}")
    # print(f"New bottom right: {new_bottom_right}")

    # Place the extracted patch at the new position
    patch[:, new_top_left[0]:new_bottom_right[0] + 1, new_top_left[1]:new_bottom_right[1] + 1] = patch_cp
    # print(f"Patch: {patch}")
    return patch


def transform_square_patch(patch, mask, patch_shape, data_shape):
    """
    Transform a batch of square patches randomly in orientation, size, and position.

    Inputs:
    patch: torch.Tensor, shape (B, C, H, W). The batch of patches to transform.
    mask: torch.Tensor, shape (H, W). The mask of the patch, same for all patches.
    patch_shape: tuple. The shape of the patch. (patch_size, patch_size)
    data_shape: tuple. The shape of the data. (B, C, H, W)

    Returns:
    patch: torch.Tensor, shape (B, C, H, W). The transformed batch of patches.
    mask: torch.Tensor, shape (H, W). The transformed mask.
    """
    # assuming data shape is square
    assert data_shape[-2] == data_shape[-1]
    image_size = data_shape[-1]
    B = data_shape[0]
    C = data_shape[2]
    
    # random rotate
    # rotate the patch where the mask is 1
    patch_ = torch.zeros((C, patch_shape[0], patch_shape[1]))
    patch_ = patch[0, :, mask == 1].clone()
    patch_ = patch_.reshape(C, patch_shape[0], patch_shape[1])
    if torch.rand(1) > 0.5:
        patch_ = torch.rot90(patch_, 1, [1, 2])
    # put the patch back to the original position
    patch[0, :, mask == 1] = patch_.reshape(patch_.shape[0], -1)
    # mask = torch.rot90(mask, num_rot, [0, 1])

    # random position
    max_x = image_size - patch_shape[0]
    max_y = image_size - patch_shape[1]
    start_x = torch.randint(max_x, (1,))
    # make sure that the patch is not near the central area as it may block the block
    central_area_coords = (image_size // 2 - 20, image_size // 2 + 20)
    # print(f"Central area coords: {central_area_coords}")
    while start_x[0] > central_area_coords[0] and start_x[0] < central_area_coords[1]:
        start_x = torch.randint(max_x, (1,))

    # move the patches to the new positions
    top_left, bottom_right = get_patch_positions(mask)
    # print(f"Top left and bottom right: {top_left}, {bottom_right}")
    for i in range(patch.shape[0]):
        patch[i] = swap_patch_position(patch[i], mask, patch_shape, top_left, bottom_right, (start_x[0], start_x[0]))
    mask = swap_mask_position(mask, patch_shape, top_left, bottom_right, (start_x[0], start_x[0]))
    # print(f"start_x: {start_x[0]}")

    # print(f"Shape of the mask and patch: {mask.shape}, {patch.shape}")

    return patch, mask

if __name__ == "__main__":
    # mask = torch.zeros((5, 5))
    # mask[1:3, 1:3] = 1
    # patch = torch.zeros((1, 1, 5 ,5))
    # patch[0, 0, 1:3, 1:3] = 3
    # print(patch)
    # patch, mask = transform_square_patch(patch, mask, (2, 2), (1, 1, 5, 5))
    # print(mask)
    # print(patch)
    # test with an image
    obs_dict = np.load("/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/obs_dict.npy")
    # get one image
    image = obs_dict[0, 0]
    image = torch.from_numpy(image)
    mask = torch.zeros((image.shape[1], image.shape[1]))
    mask[0: 15, 0: 15] = 1
    patch = torch.zeros((1, 3, 84, 84))
    patch[0, :, 0:7, 0:7] = 0.5
    patch[0, :, 8:15, 8:15] = 0.8
    patch, mask = transform_square_patch(patch, mask, (15, 15), (1, 2, 3, 84, 84))
    # overlay the patch on the image
    image = (image * (1 - mask) + mask * patch).squeeze(0)
    # save the image to check
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("image.png")
