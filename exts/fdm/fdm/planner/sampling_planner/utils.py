# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

import pypose as pp

from isaaclab.utils.math import euler_xyz_from_quat


def state_history_transformer(
    states: torch.Tensor,
    env_idx: torch.Tensor,
    history_length: int,
    exclude_index: list[int] | None = None,
):
    """transform the state history into the local robot frame

    Individual function as also used for evaluation call when the model should only do predictions.
    """
    # repeat initial state to match the state history
    initial_states = states[env_idx, 0][:, None, :7]
    initial_states_SE3 = pp.SE3(initial_states.repeat(1, history_length, 1).reshape(-1, 7))
    # transform the state history into the local robot frame
    state_history = states[env_idx, :, :7]
    state_history = pp.SE3(state_history.reshape(-1, 7))
    state_history_local = (pp.Inv(initial_states_SE3) * state_history).tensor()
    state_history_pos = state_history_local.reshape(-1, history_length, 7)[..., :2]
    state_history_yaw = euler_xyz_from_quat(state_history_local[..., [6, 3, 4, 5]])[2]
    # rotation encoded as [sin(yaw), cos(yaw)] to avoid jump in representation
    # Check: Learning with 3D rotations, a hitchhiker’s guide to SO(3), 2024, Frey et al.
    state_history_yaw = torch.stack([torch.sin(state_history_yaw), torch.cos(state_history_yaw)], dim=1)
    state_history_yaw = state_history_yaw.reshape(-1, history_length, 2)
    # get the rest of the state and potentially exclude some indices
    rest_of_state = states[env_idx, :, 7:]
    if exclude_index is not None:
        keep_idx = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        keep_idx[exclude_index] = False
        keep_idx = keep_idx[7:]
        rest_of_state = rest_of_state[..., keep_idx]
    # final state history: [N, History Length, 3 (pos) + 2 (yaw) + 1 (collision) + rest of the state]
    return torch.concatenate([state_history_pos, state_history_yaw, rest_of_state], dim=2)


def cosine_distance(r1, r2):
    """
    BS,NR
    """
    x1 = torch.stack([torch.cos(r1), torch.sin(r1)], dim=-1)
    x2 = torch.stack([torch.cos(r2), torch.sin(r2)], dim=-1)
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=-1)


def smallest_angle(yaw1, yaw2):
    """
    BS,NR
    """
    _yaw1 = yaw1 % (torch.pi * 2)
    _yaw2 = yaw2 % (torch.pi * 2)
    diff = torch.abs(_yaw1 - _yaw2)
    return torch.min(diff, torch.pi * 2 - diff)


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


@torch.jit.script
def get_non_zero_action_length(actions: torch.Tensor):
    """
    Returns the index at which all following actions are considered zero.
    Args:
        actions (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, ACTION_DIM)): Sequence of actions

    Returns:
        (torch.Tensor, dtype=torch.long, shape=(BS, NR_TRAJ, TRAJ_LENGTH)): Idx when all following actions are zero within trajectory
    """
    BS, NR_TRAJ, TRAJ_LENGTH, _ = actions.shape
    # Create a mask where the actions is set to zero

    # Example1: 1,1,1,1,0,0,0  (expected results 3)
    # Example2: 1,1,1,1,0,1,0  (expected results 5)
    actions_are_not_zero = torch.abs(actions).sum(dim=3) > 0.05

    # Flip the mask from end of trajectory to start
    # Example1: 1,1,1,1,0,0,0 -> 0,0,0,1,1,1,1
    # Example2: 1,1,1,1,0,1,0 -> 0,1,0,1,1,1,1
    actions_are_not_zero = torch.flip(actions_are_not_zero, dims=(2,))

    # This allows us to apply the cumsum operator
    # Example1: 0,0,0,1,1,1,1 -> 0,0,0,1,2,3,4
    # Example2: 0,1,0,1,1,1,1 -> 0,1,1,2,3,4,5
    action_cumsum = torch.cumsum(actions_are_not_zero, dim=2)

    # The index we are looking for is the first time the cumsum operation equals 1
    # We want to use the argmin to retrieve this point, therefore we set all zeros to the length of the trajectory
    # Example1: 0,0,0,1,2,3,4 -> 7,7,7,1,2,3,4
    # Example2: 0,1,1,2,3,4,5 -> 7,1,1,2,3,4,5
    action_cumsum[action_cumsum == 0] = TRAJ_LENGTH

    # And we penalize with increasing "timestep" to results in a unique index
    # Example1: 7,7,7,1,2,3,4 -> 7,8,9,4,6,8,10
    # Example2: 7,1,1,2,3,4,5 -> 7,2,3,5,7,9,11
    offset = torch.arange(0, TRAJ_LENGTH, device=action_cumsum.device)[None, None].repeat(BS, NR_TRAJ, 1)

    # Example1: 3
    # Example2: 1
    res = torch.argmin(offset + action_cumsum, dim=2)

    # Example1: 7 - 3 - 1 = 3
    # Example2: 7 - 1 - 1 = 5
    return TRAJ_LENGTH - res - 1


###
# Traversability filter from Elevation Mapping Cupy
# https://github.com/leggedrobotics/elevation_mapping_cupy/blob/main/elevation_mapping_cupy/script/elevation_mapping_cupy/traversability_filter.py
###


class TraversabilityFilter(nn.Module):
    def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
        self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
        self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
        self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

        # Set weights.
        self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
        self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
        self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
        self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

        self.device = device

    def __call__(self, elevation_map: torch.Tensor):
        elevation_map = elevation_map.to(self.device).to(torch.float32)

        with torch.no_grad():
            out1 = self.conv1(elevation_map)
            out2 = self.conv2(elevation_map)
            out3 = self.conv3(elevation_map)

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = torch.cat((out1, out2, out3), dim=1)
            # out = F.concat((out1, out2, out3), axis=1)
            out = self.conv_out(out.abs())
            out = torch.exp(-out)

        return out
