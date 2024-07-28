

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass

from ...utils.raisim_conversion import isaac_raisim_batched_conversion


@dataclass
class HeightScanConfig:
    """Height scan config"""

    radii: tuple[float, ...] = (0.08, 0.16, 0.26, 0.36, 0.48)
    num_points: tuple[int, ...] = (6, 8, 10, 12, 16)
    mean: float = 0.05
    std: float = 0.1
    upper_bound_z: float = 1.2
    lower_bound_z: float = -0.4


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class HeightScan:
    def __init__(
        self,
        radii: tuple[float],
        num_points: tuple[int],
        num_envs: int,
        device: str,
        mean: float = 0.05,
        std: float = 0.1,
    ):
        self.radii = radii
        self.num_points = num_points
        self.num_envs = num_envs
        self.device = device
        self.pattern = self._init_scan_pattern()

        self.mean = mean
        self.std_inv = 1.0 / std

    def _init_scan_pattern(self):
        pattern = []
        for i, r in enumerate(self.radii):
            for j in range(self.num_points[i]):
                angle = 2.0 * np.pi * j / self.num_points[i]
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = 0.0
                pattern.append([x, y, z])
        pattern = torch.tensor(pattern, dtype=torch.float).to(self.device).unsqueeze(0)
        return pattern.expand(self.num_envs, -1, -1).reshape(self.num_envs, -1, 3)

    def _rotate_pattern(self, base_quat: torch.Tensor):
        mat = quat_to_mat(base_quat)
        control_frame_x = torch.cat(
            [mat[:, 0, 0].unsqueeze(1), mat[:, 1, 0].unsqueeze(1), torch.zeros_like(mat[:, 0, 0].unsqueeze(1))], 1
        )
        control_frame_x /= control_frame_x.norm(dim=1).unsqueeze(1)
        z_axis = torch.Tensor([[0.0, 0.0, 1.0]]).to(self.device)
        control_frame_y = torch.linalg.cross(z_axis, control_frame_x)
        new_x = (
            control_frame_x[:, 0].unsqueeze(1) * self.pattern[:, :, 0]
            + control_frame_y[:, 0].unsqueeze(1) * self.pattern[:, :, 1]
        )
        new_y = (
            control_frame_x[:, 1].unsqueeze(1) * self.pattern[:, :, 0]
            + control_frame_y[:, 1].unsqueeze(1) * self.pattern[:, :, 1]
        )
        new_pattern = torch.cat([new_x.unsqueeze(2), new_y.unsqueeze(2), self.pattern[:, :, 2].unsqueeze(2)], 2)
        return new_pattern

    def get_height(
        self, base_quat: torch.Tensor, feet_pos: torch.Tensor, get_height_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """base quat : torch.Tensor [num_envs, 4]
        feet_pos: torch.Tensor [num_envs, num_feet, 3]
        """
        pattern = self._rotate_pattern(base_quat)[:, :, :2]
        pattern = pattern.unsqueeze(1).repeat(1, feet_pos.shape[1], 1, 1)
        feet_points = pattern + feet_pos[:, :, :2].unsqueeze(2)
        feet_points = feet_points.reshape(self.num_envs, -1, 2)
        heights = get_height_fn(feet_points).reshape(self.num_envs, -1, 1)
        scan = heights.reshape(self.num_envs, feet_pos.shape[1], -1) - feet_pos[:, :, 2].reshape(self.num_envs, -1, 1)
        self.scan_points = torch.cat([feet_points, heights], 2)
        return -scan.reshape(self.num_envs, -1)

    # Overwrite
    def get_obs(self, *args, use_raisim_order=False):
        heights = self.get_height(*args)
        if use_raisim_order:
            heights = isaac_raisim_batched_conversion(heights, batch_size=heights.shape[1] // 4)
        return (heights - self.mean) * self.std_inv

    def get_scan_points(self):
        return self.scan_points


if __name__ == "__main__":
    scan_gap = [0.08, 0.08, 0.10, 0.10, 0.12]
    scan_radii = [sum(scan_gap[: i + 1]) for i in range(len(scan_gap))]
    scan_radii = [0.08, 0.16, 0.26, 0.36, 0.48]
    print("r ", scan_radii)
    num_points = [6, 8, 10, 12, 16]
    num_envs = 2
    scan = HeightScan(scan_radii, num_points, num_envs, "cuda:0")
    feet_points = torch.zeros(2, 4, 3).to(scan.device)
    feet_points[:, 0, 0] += 1
    feet_points[:, 1, 0] -= 1
    feet_points[:, 2, 1] += 1
    feet_points[:, 3, 1] -= 1
    feet_points[:, 0, 2] += 2
    feet_points[:, 1, 2] -= 2
    print("feet points ", feet_points)
    quat = torch.zeros(num_envs, 4).to(scan.device)
    quat[:, 3] = 1.0
    quat[:, 2] = 0.1

    def get_height(pos):
        return torch.zeros(pos.shape[0], pos.shape[1], 1).to(scan.device) + pos[:, :, 0].unsqueeze(2)

    height_scan = scan.get_obs(quat, feet_points, get_height)
    print("height scan ", height_scan, height_scan.shape)
