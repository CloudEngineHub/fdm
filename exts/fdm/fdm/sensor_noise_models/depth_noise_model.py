

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from kornia.filters import gaussian_blur2d
from kornia.filters.sobel import spatial_gradient

if TYPE_CHECKING:
    from .depth_noise_model_cfg import DepthCameraNoiseCfg

# import glob
# import ntpath
# from PIL import Image
# from sys import argv
# from multiprocessing import Pool


def rand_perlin_2d(shape, res, device="cpu", fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    width = shape[-1]
    height = shape[-2]
    delta = (res[0] / height, res[1] / width)
    assert height % res[0] == 0, f"Perlin height resolution {res[0]} must be a multiple of height {height}"
    assert width % res[1] == 0, f"Perlin width resolution {res[1]} must be a multiple of width {width}"
    d = (height // res[0], width // res[1])
    batch_dim = shape[:-2]
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(*batch_dim, res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[..., slice1[0] : slice1[1], slice2[0] : slice2[1], :]
        .repeat_interleave(d[0], -3)
        .repeat_interleave(d[1], -2)
    )

    dot = lambda grad, shift: (
        torch.stack((grid[..., :height, :width, 0] + shift[0], grid[..., :height, :width, 1] + shift[1]), dim=-1)
        * grad[..., :height, :width, :]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:height, :width])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


class DepthCameraNoise:
    def __init__(
        self,
        cfg: DepthCameraNoiseCfg,
        device: str,
    ):
        # save config and device
        self.cfg = cfg
        self.device = device
        # convert degree into radian for angular resolution
        self.angular_resolution = torch.deg2rad(torch.tensor(self.cfg.angular_resolution_degree, device=self.device))

    """
    Call method
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Add noise to the depth image
        if img.shape[1] != 3 or img.shape[1] != 1:
            img = torch.movedim(img, -1, 1)
            dim_move = True
        else:
            dim_move = False

        img = self._down_sample(img)
        img = self._edge_noise(img)
        img = self._pixel_shuffle(img)
        img = self._distance_noise(img)

        if dim_move:
            img = torch.movedim(img, 1, -1)

        return img.clip(0, self.cfg.far_plane)

    """
    Noise Functions
    """

    def _down_sample(self, img):
        # Downsample image resolution by 2
        B, D, H, W = img.shape
        img = img.view(B, D, H // 2, 2, W // 2, 2)
        # Downsample
        img = img.min(dim=5)[0].min(dim=3)[0]
        # Upsample
        img = img.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return img

    def _edge_noise(self, img):
        # Filter out the edge in the depth image
        B, D, H, W = img.shape

        first_derivative = torch.abs(spatial_gradient(img))
        first_derivative_norm = torch.norm(first_derivative, dim=-3).reshape(B, D, H, W)

        maskout = (first_derivative_norm > self.cfg.edge_noise_thresh).float()
        # gaussian_noise = torch.randn(B, D, H, W).to(self.device)
        # maskout += gaussian_noise > 1.0
        # maskout = (maskout * noise > 0.7).float()

        noise = self._get_noise_sample((B, D, H, W))
        noise_std = self._img_to_variance(img)
        # print("noise_std ", noise_std, noise_std.shape)
        # maskout = (noise_std > self.cfg.edge_noise_thresh).float()
        # print("maskout ", maskout, maskout.shape)
        img += maskout * self.cfg.far_plane
        img += noise * noise_std
        return img

    def _pixel_shuffle(self, img):
        B, D, H, W = img.shape
        rand_nums = torch.randn(B, D, H, W, 3).to(self.device)
        x_max, y_max = W - 1, H - 1
        y = torch.floor(
            rand_nums[..., 0] * self.cfg.pixel_noise_a
            + torch.arange(0, H).view(1, 1, H, 1).repeat(B, D, 1, W).to(self.device)
        )
        x = torch.floor(
            rand_nums[..., 1] * self.cfg.pixel_noise_a
            + torch.arange(0, W).view(1, 1, 1, W).repeat(B, D, H, 1).to(self.device)
        )
        y = torch.clamp(y, 0, y_max)
        x = torch.clamp(x, 0, x_max)
        return img[
            torch.arange(0, B).view(B, 1, 1, 1).repeat(1, D, H, W),
            torch.arange(0, D).view(1, D, 1, 1).repeat(B, 1, H, W),
            y.long(),
            x.long(),
        ]

    def _distance_noise(self, img):
        B, D, H, W = img.shape
        noise = torch.randn(B, D, H, W).to(self.device)
        a = img / self.cfg.far_plane
        # img += noise * a**2
        output = gaussian_blur2d((img + noise * a**2), (3, 3), (1.5, 1.5)).reshape(B, D, H, W)
        img = img * (1 - a) + output * a
        return img

    """
    Helper functions
    """

    def _get_noise_sample(
        self,
        shape,
    ):
        """
        Parameters
        ----------
        shape [*batch_dims, height, width]
        device
        """
        noise = rand_perlin_2d(shape, self.cfg.perlin_resolution, device=self.device)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) - 0.5
        return noise

    def _get_normal_angle(self, img):
        shape = img.shape
        if len(shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        if len(shape) == 3:
            img = img.unsqueeze(1)
        # First derivative in pixel space
        first_derivative = torch.abs(spatial_gradient(img))
        first_derivative_norm = torch.norm(first_derivative, dim=-3)
        distance_between_pixels = torch.sin(self.angular_resolution) * torch.abs(img)
        theta = torch.atan2(distance_between_pixels, first_derivative_norm)

        return theta.view(shape)

    def _img_to_variance(self, img):
        theta_y = self._get_normal_angle(img)
        theta_y = torch.clip(theta_y, 0.0, torch.deg2rad(torch.tensor(80, device=self.device)))
        noise_std = (
            0.001063
            + 0.0007278 * img
            + 0.003949 * torch.square(img)
            + 0.022 * torch.pow(img, 1.5) * theta_y / torch.square(torch.pi / 2 - theta_y)
        )
        noise_std *= (img < self.cfg.far_plane).float()
        return noise_std
