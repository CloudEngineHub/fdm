# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Model adapted from:

Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation
Yunho Kim, Chanyoung Kim, Jemin Hwangbo
https://arxiv.org/abs/2204.08647

https://github.com/awesomericky/complex-env-navigation/blob/master/raisimGymTorch/env/envs/train/model.py
"""

from __future__ import annotations

import numpy as np
import prettytable
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from .fdm_model import FDMModel
from .model_base import Model

if TYPE_CHECKING:
    from .fdm_baseline_model_cfg import FDMBaselineCfg


class FDMBaseline(FDMModel):
    """
    Forward Dynamics Model Baseline

    Model from the paper:
        Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation
        Yunho Kim, Chanyoung Kim, Jemin Hwangbo
        https://arxiv.org/abs/2204.08647

    .. note::
        Inherits from the FDMModel class to have access to the same evaluation metrics. Loss and model structure are
        the same as in the original paper.
    """

    cfg: FDMBaselineCfg
    """Configuration for the FDM Baseline model."""

    def __init__(self, cfg: FDMBaselineCfg, device: str):
        Model.__init__(self, cfg, device)

        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}

        assert self.cfg.state_encoder["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.cfg.command_encoder["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.cfg.traj_predictor["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        # set parameters for evaluation
        self.param_unified_failure_prediction = self.cfg.unified_failure_prediction
        # perfect velocity tracking loss for evaluation only
        self.perfect_velocity_position_loss = nn.MSELoss()
        # set the model
        self.set_module()

        # init metrics
        self.metric_presision = BinaryPrecision(threshold=self.cfg.collision_threshold)
        self.metric_recall = BinaryRecall(threshold=self.cfg.collision_threshold)
        self.metric_accuracy = BinaryAccuracy(threshold=self.cfg.collision_threshold)

        # print number of parameters
        table = prettytable.PrettyTable(["Layer", "Parameters"])
        table.title = f"[INFO] Model Parameters (Total: {self.number_of_parameters})"
        for layer, count in self.layer_parameters.items():
            table.add_row([layer, count])
        print(table)

    def set_module(self):
        self.state_encoder = MLP(
            self.cfg.state_encoder["shape"],
            self.activation_map[self.cfg.state_encoder["activation"]],
            self.cfg.state_encoder["input"],
            self.cfg.state_encoder["output"],
            dropout=self.cfg.state_encoder["dropout"],
            batchnorm=self.cfg.state_encoder["batchnorm"],
        )
        self.command_encoder = MLP(
            self.cfg.command_encoder["shape"],
            self.activation_map[self.cfg.command_encoder["activation"]],
            self.cfg.command_encoder["input"],
            self.cfg.command_encoder["output"],
            dropout=self.cfg.command_encoder["dropout"],
            batchnorm=self.cfg.command_encoder["batchnorm"],
        )
        self.recurrence = torch.nn.LSTM(
            self.cfg.recurrence["input"],
            self.cfg.recurrence["hidden"],
            self.cfg.recurrence["layer"],
            dropout=self.cfg.recurrence["dropout"],
        )
        self.Pcol_prediction = MLP(
            self.cfg.traj_predictor["shape"],
            self.activation_map[self.cfg.traj_predictor["activation"]],
            self.cfg.traj_predictor["input"],
            self.cfg.traj_predictor["collision"]["output"],
            dropout=self.cfg.traj_predictor["dropout"],
            batchnorm=self.cfg.traj_predictor["batchnorm"],
        )
        self.coordinate_prediction = MLP(
            self.cfg.traj_predictor["shape"],
            self.activation_map[self.cfg.traj_predictor["activation"]],
            self.cfg.traj_predictor["input"],
            self.cfg.traj_predictor["coordinate"]["output"],
            dropout=self.cfg.traj_predictor["dropout"],
            batchnorm=self.cfg.traj_predictor["batchnorm"],
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, model_in: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Model
        """
        state, obs_proprioceptive, lidar_scan, command_traj = (
            model_in[0].to(self.device).contiguous(),
            model_in[1].to(self.device).contiguous(),
            model_in[2].to(self.device).contiguous(),
            model_in[3].to(self.device).contiguous(),
        )

        # Baseline does not use a sin/cos encoding for the orientation, revert it here
        state[..., 2] = torch.atan2(state[..., 2], state[..., 3])

        # construct the same input space as used in the baseline (orientation in base frame, lin vel, ang vel, lidar scan)
        # should have an overall dim of 450 (9 * history of 10 + 360 lidar)
        state_obs_proprioceptive = torch.concatenate((state[..., :3], obs_proprioceptive), dim=-1)
        state_lidar = torch.concatenate(
            (state_obs_proprioceptive.view(state_obs_proprioceptive.shape[0], -1), lidar_scan), dim=-1
        ).contiguous()
        command_traj = command_traj.contiguous()

        # switch the commands to traj_len x batch_size x command_dim
        command_traj = command_traj.permute(1, 0, 2)

        if self.cfg.cvae_retrain:
            encoded_state = self.state_encoder.architecture(state_lidar).detach()
        else:
            encoded_state = self.state_encoder.architecture(state_lidar)
        initial_cell_state = torch.broadcast_to(
            encoded_state, (self.cfg.recurrence["layer"], *encoded_state.shape)
        ).contiguous()
        initial_hidden_state = torch.zeros_like(initial_cell_state).to(self.device)

        traj_len, n_sample, single_command_dim = command_traj.shape
        command_traj = command_traj.reshape(-1, single_command_dim)
        if self.cfg.cvae_retrain:
            encoded_command = self.command_encoder.architecture(command_traj).view(traj_len, n_sample, -1).detach()
        else:
            encoded_command = self.command_encoder.architecture(command_traj).view(traj_len, n_sample, -1)

        encoded_prediction, (_, _) = self.recurrence(encoded_command, (initial_hidden_state, initial_cell_state))
        traj_len, n_sample, encoded_prediction_dim = encoded_prediction.shape
        encoded_prediction = encoded_prediction.reshape(-1, encoded_prediction_dim)
        collision_prob_traj = self.sigmoid(self.Pcol_prediction.architecture(encoded_prediction))
        collision_prob_traj = collision_prob_traj.view(
            traj_len, n_sample, self.cfg.traj_predictor["collision"]["output"]
        )

        # coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        # coordinate_traj = coordinate_traj.view(traj_len, n_sample, self.cfg.traj_predictor["coordinate"]["output"])

        delta_coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        delta_coordinate_traj = delta_coordinate_traj.view(
            traj_len, n_sample, self.cfg.traj_predictor["coordinate"]["output"]
        )

        coordinate_traj = torch.zeros(traj_len, n_sample, self.cfg.traj_predictor["coordinate"]["output"]).to(
            self.device
        )
        for i in range(traj_len):
            if i == 0:
                coordinate_traj[i, :, :] = delta_coordinate_traj[i, :, :]
            else:
                coordinate_traj[i, :, :] = coordinate_traj[i - 1, :, :] + delta_coordinate_traj[i, :, :]

        # switch back to batch_size x traj_len x command_dim
        coordinate_traj = coordinate_traj.permute(1, 0, 2)
        collision_prob_traj = collision_prob_traj.permute(1, 0, 2)

        return coordinate_traj, collision_prob_traj.squeeze(-1)

        # BS, TRAJ_LEN = state.shape[:2]

        # # Baseline does not use a sin/cos encoding for the orientation, revert it here
        # state[..., 2] = torch.atan2(state[..., 2], state[..., 3])

        # # construct the same input space as used in the baseline (orientation in base frame, lin vel, ang vel, lidar scan)
        # # should have an overall dim of 450 (9 * history of 10 + 360 lidar)
        # state_obs_proprioceptive = torch.concatenate((state[..., :3], obs_proprioceptive), dim=-1)
        # state_lidar = torch.concatenate(
        #     (state_obs_proprioceptive.reshape(BS, -1), lidar_scan), dim=-1
        # )

        # # switch the commands to traj_len x batch_size x command_dim
        # command_traj = command_traj.permute(1, 0, 2).contiguous()

        # if self.cfg.cvae_retrain:
        #     encoded_state = self.state_encoder.architecture(state_lidar).detach()
        # else:
        #     encoded_state = self.state_encoder.architecture(state_lidar)
        # initial_cell_state = torch.broadcast_to(
        #     encoded_state, (self.cfg.recurrence["layer"], *encoded_state.shape)
        # ).contiguous()
        # initial_hidden_state = torch.zeros_like(initial_cell_state).to(self.device)

        # traj_len, n_sample, single_command_dim = command_traj.shape
        # command_traj = command_traj.reshape(-1, single_command_dim)
        # if self.cfg.cvae_retrain:
        #     encoded_command = self.command_encoder.architecture(command_traj).reshape(traj_len, n_sample, -1).detach()
        # else:
        #     encoded_command = self.command_encoder.architecture(command_traj).reshape(traj_len, n_sample, -1)

        # encoded_prediction, (_, _) = self.recurrence(encoded_command, (initial_hidden_state, initial_cell_state))
        # traj_len, n_sample, encoded_prediction_dim = encoded_prediction.shape
        # encoded_prediction = encoded_prediction.reshape(-1, encoded_prediction_dim)
        # collision_prob_traj = self.sigmoid(self.Pcol_prediction.architecture(encoded_prediction))
        # collision_prob_traj = collision_prob_traj.reshape(
        #     traj_len, n_sample, self.cfg.traj_predictor["collision"]["output"]
        # )

        # # coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        # # coordinate_traj = coordinate_traj.view(traj_len, n_sample, self.cfg.traj_predictor["coordinate"]["output"])

        # delta_coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        # delta_coordinate_traj = delta_coordinate_traj.reshape(
        #     traj_len, n_sample, self.cfg.traj_predictor["coordinate"]["output"]
        # )

        # coordinate_traj = torch.zeros(traj_len, n_sample, self.cfg.traj_predictor["coordinate"]["output"]).to(
        #     self.device
        # ).contiguous()
        # for i in range(traj_len):
        #     if i == 0:
        #         coordinate_traj[i, :, :] = delta_coordinate_traj[i, :, :]
        #     else:
        #         coordinate_traj[i, :, :] = coordinate_traj[i - 1, :, :] + delta_coordinate_traj[i, :, :]

        # # switch back to batch_size x traj_len x command_dim
        # coordinate_traj = coordinate_traj.permute(1, 0, 2).contiguous()
        # collision_prob_traj = collision_prob_traj.permute(1, 0, 2).contiguous()

    def loss(
        self,
        model_out: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mode: str = "train",
        suffix: str = "",
    ) -> tuple[torch.Tensor, dict]:

        # extract predictions and targets
        predicted_coordinates, predicted_P_cols = model_out[0], model_out[1]
        coordinates_batch, P_cols_batch = (
            target[..., :2].to(self.device),
            target[..., 4].to(self.device),
        )

        # Collision probability loss (CLE)
        if self.cfg.unified_failure_prediction:
            P_cols_batch = torch.max(P_cols_batch, dim=1).values
            predicted_P_cols = torch.max(predicted_P_cols, dim=1).values

        P_col_loss = -(
            P_cols_batch * torch.log(predicted_P_cols + 1e-6)
            + (1 - P_cols_batch) * torch.log(1 - predicted_P_cols + 1e-6)
        )
        P_col_loss = torch.sum(P_col_loss, dim=0).mean()

        # Coordinate loss (MSE)
        coordinate_loss = torch.sum(torch.sum((predicted_coordinates - coordinates_batch).pow(2), dim=-1), dim=0).mean()

        # Square root coordinate loss (Just for logging)
        square_root_coordinate_loss = torch.sum(
            torch.sqrt(torch.sum((predicted_coordinates - coordinates_batch).pow(2), dim=-1)), dim=0
        ).mean()

        position_loss = coordinate_loss * self.cfg.loss_weights["coordinate"]
        collision_prob_loss = P_col_loss * self.cfg.loss_weights["collision"]
        loss = collision_prob_loss + position_loss

        # get meta
        meta = {
            f"{mode}{suffix} Loss [Batch]": loss.item(),
            f"{mode}{suffix} Position Loss [Batch]": position_loss.item(),
            f"{mode}{suffix} Collision Loss [Batch]": collision_prob_loss.item(),
            f"{mode}{suffix} Square Root Position Loss [Batch]": square_root_coordinate_loss.item(),
        }

        return loss, meta

    def eval_metrics(
        self,
        model_out: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        eval_in: torch.Tensor,
        meta: dict | None = None,
        mode: str = "train",
        suffix: str = "",
    ) -> dict:
        pred_state_traj, pred_collision_prob_traj = model_out[0], model_out[1]
        target_state_traj, target_collision_state_traj = target[..., :2].to(self.device), target[..., 4].to(self.device)
        # get the indices of the samples in collision
        collision_idx = torch.any(target_collision_state_traj == 1, dim=1)

        if meta is None:
            meta = {}

        # compare to perfect velocity tracking error when eval mode
        meta = self._eval_perf_velocity(
            meta, mode, eval_in, target_state_traj, pred_state_traj, pred_collision_prob_traj, collision_idx, suffix
        )

        # Position offsets
        meta = self._eval_position_offsets(
            meta, mode, pred_state_traj, pred_collision_prob_traj, target_state_traj, collision_idx, suffix
        )

        # collision metrics
        if self.param_unified_failure_prediction:
            pred_collision_prob_traj = torch.max(pred_collision_prob_traj, dim=-1).values
        meta = self._eval_collision_pred(meta, mode, pred_collision_prob_traj, target_collision_state_traj, suffix)

        return meta

    ###
    # Override unused methods
    ###

    def set_acceleration_limits(self, acceleration_limits: torch.Tensor):
        # check for each acceleration if a larger value has been observed in each of the elements of the tensor
        pass

    def set_velocity_limits(self, velocity_limits: torch.Tensor):
        pass

    def set_hard_contact_obs_limits(self, min_hard_contact_obs: torch.Tensor, max_hard_contact_obs: torch.Tensor):
        pass


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, dropout=0.0, batchnorm=False):
        super().__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            if batchnorm:
                modules.append(nn.BatchNorm1d(shape[idx + 1]))
            modules.append(self.activation_fn())
            if dropout != 0.0:
                modules.append(nn.Dropout(dropout))
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
