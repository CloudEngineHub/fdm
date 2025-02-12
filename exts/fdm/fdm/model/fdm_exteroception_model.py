# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import numpy as np
import prettytable
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any

import wandb

from ..sensor_noise_models import DepthCameraNoise
from .model_base import Model

if TYPE_CHECKING:
    from .fdm_exteroception_model_cfg import FDMExteroceptionModelCfg


class FDMExteroceptionModel(Model):
    cfg: FDMExteroceptionModelCfg
    """Model config class"""

    def __init__(self, cfg: FDMExteroceptionModelCfg, device: str):
        super().__init__(cfg, device)

        # build layers
        self.obs_exteroceptive_encoder = self._construct_layer(self.cfg.obs_exteroceptive_encoder)
        self.add_obs_exteroceptive_encoder = self._construct_layer(self.cfg.add_obs_exteroceptive_encoder)

        self.image_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(256, 2, 2)),
            nn.UpsamplingBilinear2d(size=(3, 6)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # BS, 128, 6, 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # BS, 64, 12, 24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # BS, 32, 24, 48
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(43, 78)),  # BS, 32, 43, 78  (not exactly factor 2)
            nn.ConvTranspose2d(
                32,
                1 if self.cfg.obs_exteroceptive_encoder.individual_channel_encoding else 3,
                kernel_size=7,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # BS, 32, 90, 160
        )

        self.height_map_decoder = nn.Sequential(
            nn.Linear(1024 * 3 + self.cfg.add_obs_exteroceptive_encoder.output, 1024),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(256, 2, 2)),
            nn.UpsamplingBilinear2d(size=(4, 3)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # BS, 128, 8, 6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # BS, 64, 16, 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # BS, 32, 32, 24
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 23)),  # BS, 32, 30, 23
            nn.ConvTranspose2d(
                32,
                1 if self.cfg.obs_exteroceptive_encoder.individual_channel_encoding else 3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # BS, 32, 60, 46
        )

        # init loss functions
        self.height_map_loss = nn.MSELoss()
        self.depth_reconstruction_loss = nn.MSELoss()

        # init depth camera noise model for image noising
        if self.cfg.depth_camera_noise is not None:
            self.depth_camera_noise = DepthCameraNoise(self.cfg.depth_camera_noise, self.device)
        else:
            self.depth_camera_noise = None

        # image save interval -- number of batches between 1 image is saved
        self.image_save_interval = 20

        # print number of parameters
        table = prettytable.PrettyTable(["Layer", "Parameters"])
        table.title = f"[INFO] Model Parameters (Total: {self.number_of_parameters})"
        for layer, count in self.layer_parameters.items():
            table.add_row([layer, count])
        print(table)

    """
    Forward function of the dynamics model
    """

    def forward(self, model_in: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model

        Args:
            - state: State of the robots. Shape (batch_size, state_dim)
            - command_traj: Commands along the recorded trajectory. Shape (batch_size, traj_len, command_dim)

        Returns:
            - coordinate: Coordinate of the robot along the trajectory. Shape (batch_size, traj_len, 2)
            - collision_prob_traj: Probability of collision along the trajectory. Shape (batch_size, traj_len) if
                                   not cfg.unified_failure_prediction else shape (batch_size).
        """
        obs_extereoceptive, add_obs_exteroceptive = (
            model_in[2].to(self.device),
            model_in[4].to(self.device)[
                :, : -int(self.cfg.target_height_map_size[0] * self.cfg.target_height_map_size[1])
            ],
        )

        ###
        # Noisify images
        ###
        if self.depth_camera_noise is not None:
            obs_extereoceptive = self.depth_camera_noise(obs_extereoceptive)

        ###
        # Encode inputs
        ###

        # encode exteroceptive observations
        encoded_obs_exteroceptive = self.obs_exteroceptive_encoder(obs_extereoceptive)
        encoded_add_obs_exteroceptive = self.add_obs_exteroceptive_encoder(add_obs_exteroceptive)

        ###
        # Predict
        ###
        # depth reconstruction
        if self.cfg.obs_exteroceptive_encoder.individual_channel_encoding:
            decoding = [
                self.image_decoder(encoded_obs_exteroceptive[:, i * 1024 : (i + 1) * 1024])
                for i in range(obs_extereoceptive.shape[-1])
            ]
            depth_reconstruction = torch.concat(decoding, dim=1)
        else:
            depth_reconstruction = self.image_decoder(encoded_obs_exteroceptive)

        if depth_reconstruction.shape[-1] != obs_extereoceptive.shape[-1]:
            depth_reconstruction = torch.movedim(depth_reconstruction, 1, -1)

        # height map
        height_map = self.height_map_decoder(
            torch.concatenate([encoded_obs_exteroceptive, encoded_add_obs_exteroceptive], dim=-1)
        ).squeeze(1)

        return depth_reconstruction, height_map

    def loss(
        self,
        model_in: tuple[torch.Tensor, torch.Tensor],
        model_out: tuple[torch.Tensor, torch.Tensor],
        mode: str = "train",
        suffix: str = "",
    ) -> tuple[torch.Tensor, dict]:
        """Network loss function as a combintation of the collision probability loss and the coordinate loss"""
        # extract predictions and targets
        target_depth_reconstruction, target_height_map = (
            model_in[2].to(self.device),
            model_in[4].to(self.device)[
                :, -int(self.cfg.target_height_map_size[0] * self.cfg.target_height_map_size[1]) :
            ],
        )
        pred_depth_reconstruction, pred_height_map = model_out[0], model_out[1]

        # resize target height map
        # the map is storted that the x-values are running first and in the second dim the y values are running
        # i.e. x[0, 0] will have fixed y and all x values, then x[0, 1] will have fixed y+1 and all x values
        # thus target has shape (BS, x.shape, y.shape)
        target_height_map = target_height_map.view(
            target_height_map.shape[0], self.cfg.target_height_map_size[1], self.cfg.target_height_map_size[0]
        )

        # calculate losses
        # images are shape (BS, H, W, C)
        depth_reconstruction_loss = self.depth_reconstruction_loss(
            pred_depth_reconstruction, target_depth_reconstruction
        )
        height_map_loss = self.height_map_loss(pred_height_map, target_height_map)

        # combine losses
        loss = (
            depth_reconstruction_loss * self.cfg.loss_weights["depth_reconstruction"]
            + height_map_loss * self.cfg.loss_weights["height_map"]
        )

        # save meta data
        meta = {
            f"{mode}{suffix} Loss [Batch]": loss.item(),
            f"{mode}{suffix} Depth Reconstruction [Batch]": depth_reconstruction_loss.item(),
            f"{mode}{suffix} Height Map [Batch]": height_map_loss.item(),
        }

        if mode == "eval":
            # get a random image index
            idx = torch.randint(0, target_depth_reconstruction.shape[0], (1,)).item()
            max_img_depth = torch.max(
                target_depth_reconstruction[idx].max(), pred_depth_reconstruction[idx].max()
            ).item()
            with contextlib.suppress(wandb.errors.Error):
                wandb.log({
                    "pred_images": [
                        wandb.Image(
                            (pred_depth_reconstruction[idx, :, :, img_idx].cpu().numpy() / max_img_depth * 255).astype(
                                np.uint8
                            )
                        )
                        for img_idx in range(3)
                    ],
                    "target_images": [
                        wandb.Image(
                            (
                                target_depth_reconstruction[idx, :, :, img_idx].cpu().numpy() / max_img_depth * 255
                            ).astype(np.uint8)
                        )
                        for img_idx in range(3)
                    ],
                    "pred_height_map": wandb.Image(pred_height_map[idx].cpu().numpy()),
                    "target_height_map": wandb.Image(target_height_map[idx].cpu().numpy()),
                })

        return loss, meta

    ###
    # Reimplementations to pass input to loss function directly and skip metric compute
    ###

    def evaluate(
        self,
        model_in: torch.Tensor | tuple[torch.Tensor, ...],
        target: None | torch.Tensor | tuple[torch.Tensor, ...] = None,
        eval_in: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Reimplemenation."""
        self.eval()
        with torch.inference_mode():
            model_out = self.forward(model_in)
            loss, meta = self.loss(model_in, model_out, "eval")
        return loss.item(), meta

    def update(
        self,
        model_in: torch.Tensor | tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        target: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        eval_in: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Reimplemenation."""
        self.train()
        optimizer.zero_grad()
        model_out = self.forward(model_in)
        loss, meta = self.loss(model_in, model_out)
        loss.backward()
        if self.cfg.max_grad_norm:
            nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
        optimizer.step()
        return loss.item(), meta

    def eval_metrics(
        self,
        model_out: torch.Tensor | tuple[torch.Tensor, ...],
        target: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        eval_in: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        meta: dict[str, Any] | None = None,
        mode: str = "train",
        suffix: str = "",
    ) -> dict[str, Any]:
        pass

    def set_velocity_limits(self, velocity_limits: torch.Tensor):
        pass
