# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
These files are copied from isaaclab https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/utils/math.py
"""

import torch
from typing import Literal, Tuple

import pypose as pp


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (N, 4).
        v: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_torch = torch.tensor(mag, device=other.device, dtype=torch.float).repeat(other.shape[0])
    return torch.abs(mag_torch) * torch.sign(other)


@torch.jit.script
def euler_xyz_from_quat(quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.clone().view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector only around the yaw-direction.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (N, 4).
        vec: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return normalize(quat_conjugate(q))


"""
Planning helpers
"""


@torch.jit.script
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


@torch.jit.script
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


def cosine_distance(yaw1, yaw2):
    """
    BS,NR
    """
    x1 = torch.stack([torch.cos(yaw1), torch.sin(yaw1)], dim=-1)
    x2 = torch.stack([torch.cos(yaw2), torch.sin(yaw2)], dim=-1)
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=-1)


def smallest_angle(yaw1, yaw2):
    """
    BS,NR
    """
    _yaw1 = yaw1 % (torch.pi * 2)
    _yaw2 = yaw2 % (torch.pi * 2)
    diff = torch.abs(_yaw1 - _yaw2)
    return torch.min(diff, torch.pi * 2 - diff)


"""
Grid Pattern for the height scan
"""


def grid_pattern(
    size: Tuple[float, float],
    resolution: float,
    ordering: Literal["xy", "yx"],
    direction: Tuple[float, float, float],
    offset: Tuple[float, float, float],
    device: str,
) -> torch.Tensor:
    """A regular grid pattern for ray casting.

    The grid pattern is made from rays that are parallel to each other. They span a 2D grid in the sensor's
    local coordinates from ``(-length/2, -width/2)`` to ``(length/2, width/2)``, which is defined
    by the ``size = (length, width)`` and ``resolution`` parameters in the config.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.

    Raises:
        ValueError: If the ordering is not "xy" or "yx".
        ValueError: If the resolution is less than or equal to 0.
    """
    # check valid arguments
    if ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{ordering}'.")
    if resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{resolution}'.")

    # resolve mesh grid indexing (note: torch meshgrid is different from numpy meshgrid)
    # check: https://github.com/pytorch/pytorch/issues/15301
    indexing = ordering if ordering == "xy" else "ij"
    # define grid pattern
    x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution, device=device)
    y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)

    # store into ray starts
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    # apply offset
    ray_starts += torch.tensor(offset, device=device)

    return ray_starts


"""
State history transformer
"""


def state_history_transformer(
    states: torch.Tensor,
    history_length: int,
):
    """transform the state history into the local robot frame

    Individual function as also used for evaluation call when the model should only do predictions.
    """
    # repeat initial state to match the state history
    initial_states_SE3 = pp.SE3(states[0, :7].unsqueeze(0).repeat(history_length, 1))
    # transform the state history into the local robot frame
    state_history = pp.SE3(states[:, :7])
    state_history_local = (pp.Inv(initial_states_SE3) * state_history).tensor()
    state_history_pos = state_history_local[:, :2]
    state_history_yaw = euler_xyz_from_quat(state_history_local[:, 3:7])[2]
    # rotation encoded as [sin(yaw), cos(yaw)] to avoid jump in representation
    # Check: Learning with 3D rotations, a hitchhiker’s guide to SO(3), 2024, Frey et al.
    state_history_yaw = torch.stack([torch.sin(state_history_yaw), torch.cos(state_history_yaw)], dim=1)
    # final state history: [N, History Length, 3 (pos) + 2 (yaw) + 1 (collision) + rest of the state]
    return torch.concatenate([state_history_pos, state_history_yaw, states[:, 7:]], dim=-1)
