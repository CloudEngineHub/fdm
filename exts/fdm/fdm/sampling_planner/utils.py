

import torch


def cosine_distance(r1, r2):
    """
    BS,NR
    """
    x1 = torch.stack([torch.cos(r1), torch.sin(r1)], dim=-1)
    x2 = torch.stack([torch.cos(r2), torch.sin(r2)], dim=-1)
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=-1)


def get_se2(state):
    """
    yaw (torch.tensor, shape (BS, NR_TRAJ, TRAJ_LENGTH, 3): assumes input being X,Y,YAW
    """
    dim = len(state.shape)
    c_vec1 = torch.stack(
        [torch.cos(state[..., 2]), torch.sin(state[..., 2]), torch.zeros_like(state[..., 0])], dim=dim - 1
    )
    c_vec2 = torch.stack(
        [-torch.sin(state[..., 2]), torch.cos(state[..., 2]), torch.zeros_like(state[..., 0])], dim=dim - 1
    )
    c_vec3 = torch.stack([state[..., 0], state[..., 1], torch.ones_like(state[..., 0])], dim=dim - 1)
    return torch.stack([c_vec1, c_vec2, c_vec3], dim=dim)


def get_x_y_yaw(se2):
    """
    se2 (torch.tensor, shape (BS, NR_TRAJ, TRAJ_LENGTH, 3, 3): assumes input being a 3x3 se2 matrix
    """
    dim = len(se2.shape)
    return torch.stack([se2[..., 0, 2], se2[..., 1, 2], torch.atan2(se2[..., 1, 0], se2[..., 0, 0])], dim=dim - 2)
