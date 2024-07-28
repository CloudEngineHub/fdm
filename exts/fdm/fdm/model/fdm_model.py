

from __future__ import annotations

import math
import prettytable
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from .model_base import Model
from .model_base_cfg import BaseModelCfg
from .utils import EmpiricalNormalization, L2Loss

if TYPE_CHECKING:
    from .fdm_model_cfg import FDMBaseModelCfg, FDMModelVelocitySingleStepHeightAdjustCfg


class FDMModel(Model):
    cfg: FDMBaseModelCfg
    """Model config class"""

    def __init__(self, cfg: FDMBaseModelCfg, device: str):
        super().__init__(cfg, device)

        # build encoder layers
        self.state_obs_proprioceptive_encoder = self._construct_layer(self.cfg.state_obs_proprioception_encoder)
        self.obs_exteroceptive_encoder = self._construct_layer(self.cfg.obs_exteroceptive_encoder)
        if self.cfg.action_encoder is not None:
            self.action_encoder = self._construct_layer(self.cfg.action_encoder)
        else:
            self.action_encoder = None
        if self.cfg.add_obs_exteroceptive_encoder is not None:
            self.add_obs_exteroceptive_encoder = self._construct_layer(self.cfg.add_obs_exteroceptive_encoder)
        else:
            self.add_obs_exteroceptive_encoder = None

        # build prediction layers
        self.recurrence = self._construct_layer(self.cfg.recurrence)
        self.state_predictor = self._construct_layer(self.cfg.state_predictor)
        self.collision_predictor = self._construct_layer(self.cfg.collision_predictor)
        self.energy_predictor = self._construct_layer(self.cfg.energy_predictor)
        self.friction_predictor = self._construct_layer(self.cfg.friction_predictor)
        self.sigmoid = nn.Sigmoid()

        # include empirical normalizer for the proprioceptive observations
        self.proprioceptive_normalizer = EmpiricalNormalization(self.cfg.empirical_normalization_dim)

        # init loss functions
        # NOTE: this is not the same loss as used by (Kim et al., 2022), to generate the same loss it is required to
        # use nn.BCELoss(reduction="sum") and divide the loss by the prediction horizon
        # P_col_loss = - (P_cols_batch * torch.log(predicted_P_cols + 1e-6) + (1 - P_cols_batch) * torch.log(1 - predicted_P_cols + 1e-6))
        # torch.sum(P_col_loss, dim=0).mean()
        self.probability_loss = nn.BCELoss()

        # NOTE: this is not the same loss as used by (Kim et al., 2022), to generate the same loss it is required to
        # use nn.MSELoss(reduction="sum") and divide the loss by the prediction horizon
        # original: torch.sum(torch.sum((predicted_coordinates - coordinates_batch).pow(2), dim=-1), dim=0).mean()
        # NOTE: here we use reduction="sum" to get access to the error in the different distances of the horizon
        if self.cfg.pos_loss_norm == "mse":
            self.position_loss = nn.MSELoss()
        elif self.cfg.pos_loss_norm == "l2":
            self.position_loss = L2Loss
        elif self.cfg.pos_loss_norm == "l1":
            self.position_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown position loss norm: {self.cfg.pos_loss_norm}")
        self.heading_loss = nn.MSELoss()

        # perfect velocity tracking loss for evaluation only
        self.perfect_velocity_position_loss = nn.MSELoss()

        # stop loss
        self.stop_loss = nn.MSELoss()

        # energy loss
        self.energy_loss = nn.MSELoss()

        # init metrics
        self.metric_presision = BinaryPrecision(threshold=0.5)
        self.metric_recall = BinaryRecall(threshold=0.5)
        self.metric_accuracy = BinaryAccuracy(threshold=0.5)

        # init velocity and acceleration limit buffer --> filled by maximum oberserved simulation values
        self.acceleration_limits = torch.zeros(3, device=self.device)
        self.velocity_limits = torch.zeros(3, device=self.device)

        # learning progress
        self._learning_progress_step = 1.0
        self._update_step = 0

        # print number of parameters
        table = prettytable.PrettyTable(["Layer", "Parameters"])
        table.title = f"[INFO] Model Parameters (Total: {self.number_of_parameters})"
        for layer, count in self.layer_parameters.items():
            table.add_row([layer, count])
        print(table)

    """
    Update physical limits
    """

    def set_acceleration_limits(self, acceleration_limits: torch.Tensor):
        # check for each acceleration if a larger value has been observed in each of the elements of the tensor
        self.acceleration_limits = torch.maximum(self.acceleration_limits, acceleration_limits.to(self.device))

    def set_velocity_limits(self, velocity_limits: torch.Tensor):
        self.velocity_limits = torch.maximum(self.velocity_limits, velocity_limits.to(self.device))

    """
    Learning progress update
    """

    def set_learning_progress_step(self, learning_progress_step: float):
        """Set the learning progress step for the model. This can be used to scale the loss during training."""
        self._learning_progress_step = learning_progress_step

    """
    Forward function of the dynamics model
    """

    def forward(self, model_in: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        state, obs_proprioceptive, obs_extereoceptive, actions, add_obs_exteroceptive = (
            model_in[0].to(self.device),
            model_in[1].to(self.device),
            model_in[2].to(self.device),
            model_in[3].to(self.device),
            model_in[4].to(self.device),
        )

        ###
        # Encode inputs
        ###

        # encode state and proprioceptive observations
        if isinstance(self.cfg.state_obs_proprioception_encoder, BaseModelCfg.MLPConfig):
            # MLP - only work with history equal to 1
            encoded_state_obs_proprioceptive = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
        else:
            # recurrent - work with history > 1
            # TODO: for GRU the same, for S4RNN check if the same as the hidden
            encoded_state_obs_proprioceptive, _ = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )

        # encode exteroceptive observations
        encoded_obs_exteroceptive = self.obs_exteroceptive_encoder(obs_extereoceptive)
        # encode action trajectory if encoder is given
        batch_size, traj_len, single_action_dim = actions.shape
        if self.action_encoder is None:
            encoded_actions = actions
        else:
            actions = actions.view(-1, single_action_dim)
            encoded_actions = self.action_encoder.forward(actions).view(batch_size, traj_len, -1)

        # concatenate last state and proprioceptive observation with exteroceptive observation
        if self.add_obs_exteroceptive_encoder is not None:
            encoded_add_obs_exteroceptive = self.add_obs_exteroceptive_encoder(add_obs_exteroceptive)
            encoded_state_obs = torch.concatenate(
                [encoded_state_obs_proprioceptive[:, -1, :], encoded_obs_exteroceptive, encoded_add_obs_exteroceptive],
                dim=1,
            )
        else:
            encoded_state_obs = torch.concatenate(
                [encoded_state_obs_proprioceptive[:, -1, :], encoded_obs_exteroceptive], dim=1
            )

        ###
        # Predict
        ###

        if isinstance(self.cfg.recurrence, BaseModelCfg.MLPConfig):
            forward_predict_state_obs = self.recurrence(
                torch.concatenate([encoded_actions.flatten(start_dim=1), encoded_state_obs], dim=1)
            )
        else:
            # set hidden state
            initial_hidden_state = torch.broadcast_to(
                encoded_state_obs, (self.cfg.recurrence.num_layers, *encoded_state_obs.shape)
            ).contiguous()

            # recurrent forward predict encoding of state and obsverations given the commands
            forward_predict_state_obs, _ = self.recurrence(encoded_actions, initial_hidden_state)
            forward_predict_state_obs = forward_predict_state_obs.reshape(-1, forward_predict_state_obs.shape[2])

        # predict the probability of collision along the trajectory
        collision_prob_traj = self.sigmoid(self.collision_predictor.forward(forward_predict_state_obs))
        collision_prob_traj = collision_prob_traj.view(batch_size, traj_len)
        if self.cfg.unified_failure_prediction:
            collision_prob_traj = torch.max(collision_prob_traj, dim=-1)[0]

        # energy prediction
        energy_traj = self.energy_predictor(forward_predict_state_obs)
        energy_traj = energy_traj.view(batch_size, traj_len, -1)

        # predict the state transitions between consecutive commands in robot frame
        rel_state_transitions = self.state_predictor.forward(forward_predict_state_obs)
        rel_state_transitions = rel_state_transitions.view(batch_size, traj_len, -1)

        # add relative transitions to previous ones to get absolute transitions
        state_traj = torch.cumsum(rel_state_transitions, dim=1)

        return state_traj, collision_prob_traj, energy_traj

    def loss(
        self,
        model_out: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mode: str = "train",
        suffix: str = "",
    ) -> tuple[torch.Tensor, dict]:
        """Network loss function as a combintation of the collision probability loss and the coordinate loss"""
        # extract predictions and targets
        pred_state_traj, pred_collision_prob_traj, energy_traj = model_out[0], model_out[1], model_out[2]
        target_state_traj, target_collision_state_traj, target_energy_traj = (
            target[..., :4].to(self.device),
            target[..., 4].to(self.device),
            target[..., 5].to(self.device),
        )

        # stop loss - do not move when collision has happenend
        # note: has to be called before the collision probability loss, because there can be unified
        stop_loss = self.stop_loss(
            pred_state_traj[target_collision_state_traj == 1], target_state_traj[target_collision_state_traj == 1]
        )
        stop_loss = torch.tensor(0.0, device=self.device) if torch.isnan(stop_loss) else stop_loss

        # Collision probability loss (CLE)
        if self.cfg.unified_failure_prediction:
            target_collision_state_traj = torch.max(target_collision_state_traj, dim=-1)[0]
        collision_prob_loss = self.probability_loss(pred_collision_prob_traj, target_collision_state_traj)

        # Position loss (MSE)
        if self.cfg.weight_inverse_distance:
            position_loss_list = [
                (
                    (pred_state_traj[:, idx, :2] - target_state_traj[:, idx, :2]) ** 2
                    / (torch.norm(target_state_traj[:, idx, :2], dim=-1) + 1e-6).unsqueeze(1)
                ).mean()
                * torch.norm(target_state_traj[:, idx, :2], dim=-1).mean()
                for idx in range(pred_state_traj.shape[1])
            ]
        else:
            position_loss_list = [
                self.position_loss(pred_state_traj[:, idx, :2], target_state_traj[:, idx, :2])
                for idx in range(pred_state_traj.shape[1])
            ]
        position_loss = torch.sum(torch.stack(position_loss_list), dim=0)  # / pred_state_traj.shape[1]

        # Heading loss (MSE)
        heading_loss_list = [
            self.heading_loss(pred_state_traj[:, idx, 2:], target_state_traj[:, idx, 2:])
            for idx in range(pred_state_traj.shape[1])
        ]
        heading_loss = torch.sum(torch.stack(heading_loss_list), dim=0)  # / pred_state_traj.shape[1]

        # get velocity
        position_delta = torch.zeros((pred_state_traj.shape[0], pred_state_traj.shape[1], 3), device=self.device)
        # convert heading representation to radians
        pred_heading_change = torch.atan2(pred_state_traj[:, :, 2], pred_state_traj[:, :, 3])
        position_delta[:, 1:, :2] = pred_state_traj[:, 1:, :2] - pred_state_traj[:, :-1, :2]
        position_delta[:, 1:, 2] = pred_heading_change[:, 1:] - pred_heading_change[:, :-1]
        position_delta[:, 0, :2] = pred_state_traj[:, 0, :2]
        position_delta[:, 0, 2] = pred_heading_change[:, 0]
        velocity = position_delta / self.cfg.command_timestep

        # Velocity loss (sum of violations)
        velocity_loss = torch.abs(velocity) - self.velocity_limits
        velocity_loss = torch.sum(velocity_loss.clip(min=0.0)) / pred_state_traj.shape[0]

        # get acceleration
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / self.cfg.command_timestep
        # Acceleration loss (sum of violations)
        acceleration_loss = torch.abs(acceleration) - self.acceleration_limits
        acceleration_loss = torch.sum(acceleration_loss.clip(min=0.0))

        # Energy loss (MSE)
        energy_loss = self.energy_loss(energy_traj.squeeze(-1), target_energy_traj)

        # scale losses
        collision_prob_loss *= self.cfg.loss_weights["collision"]
        position_loss *= self.cfg.loss_weights["position"]
        heading_loss *= self.cfg.loss_weights["heading"]
        velocity_loss *= self.cfg.loss_weights["velocity"]
        acceleration_loss *= self.cfg.loss_weights["acceleration"]
        stop_loss *= self.cfg.loss_weights["stop"]
        energy_loss *= self.cfg.loss_weights["energy"]

        # scale losses by learning progress if enabled
        collision_prob_loss *= (
            self._learning_progress_step * self._update_step if self.cfg.progress_scaling["collision"] else 1.0
        )
        position_loss *= (
            self._learning_progress_step * self._update_step if self.cfg.progress_scaling["position"] else 1.0
        )
        heading_loss *= (
            self._learning_progress_step * self._update_step if self.cfg.progress_scaling["heading"] else 1.0
        )
        velocity_loss *= (
            self._learning_progress_step * self._update_step if self.cfg.progress_scaling["velocity"] else 1.0
        )
        acceleration_loss *= (
            self._learning_progress_step * self._update_step if self.cfg.progress_scaling["acceleration"] else 1.0
        )
        stop_loss *= self._learning_progress_step * self._update_step if self.cfg.progress_scaling["stop"] else 1.0
        energy_loss *= self._learning_progress_step * self._update_step if self.cfg.progress_scaling["energy"] else 1.0

        # combine losses
        loss = (
            collision_prob_loss
            + position_loss
            + heading_loss
            + velocity_loss
            + acceleration_loss
            + stop_loss
            + energy_loss
        )

        # save meta data
        meta = {
            f"{mode}{suffix} Loss [Batch]": loss.item(),
            f"{mode}{suffix} Collision Loss [Batch]": collision_prob_loss.item(),
            f"{mode}{suffix} Position Loss [Batch]": position_loss.item(),
            f"{mode}{suffix} Heading Loss [Batch]": heading_loss.item(),
            f"{mode}{suffix} Velocity Loss [Batch]": velocity_loss.item(),
            f"{mode}{suffix} Acceleration Loss [Batch]": acceleration_loss.item(),
            f"{mode}{suffix} Stop Loss [Batch]": stop_loss.item(),
            f"{mode}{suffix} Energy Loss [Batch]": energy_loss.item(),
        }
        if mode != "test":  # avoid overflow of information
            [
                meta.update({f"{mode}{suffix} Position Loss Horizon {idx} [Batch]": position_loss_list[idx].item()})
                for idx in range(0, len(position_loss_list), int(len(position_loss_list) / 5))
            ]
            [
                meta.update({f"{mode}{suffix} Heading Loss Horizon {idx} [Batch]": heading_loss_list[idx].item()})
                for idx in range(0, len(heading_loss_list), int(len(heading_loss_list) / 5))
            ]
        meta.update({f"{mode}{suffix} Position Loss Horizon Last [Batch]": position_loss_list[-1].item()})
        meta.update({f"{mode}{suffix} Heading Loss Horizon Last [Batch]": heading_loss_list[-1].item()})

        if any(list(self.cfg.progress_scaling.values())) and mode == "train":
            meta["Learning Progress Scaling"] = self._learning_progress_step * (self._update_step - 1)

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
        pred_state_traj, pred_collision_prob_traj, _ = model_out[0], model_out[1], model_out[2]
        target_state_traj, target_collision_state_traj = target[..., :4].to(self.device), target[..., 4].to(self.device)

        if meta is None:
            meta = {}

        # compare to perfect velocity tracking error when eval mode
        if (mode == "eval" or mode == "test") and "eval Position Loss [Batch]" in meta:
            if isinstance(eval_in, (tuple, list)):
                eval_in = eval_in[0].to(self.device)
            else:
                eval_in = eval_in.to(self.device)
            constant_perf_vel_loss = self.perfect_velocity_position_loss(eval_in, target_state_traj).item()
            model_loss = self.perfect_velocity_position_loss(pred_state_traj, target_state_traj).item()
            meta[f"{mode}{suffix} Perfect Velocity Position Loss non-collision samples [Batch]"] = (
                model_loss / constant_perf_vel_loss
            )

        # Heading error in degrees
        meta[f"{mode}{suffix} Heading Degree Error [Batch]"] = torch.rad2deg(
            torch.mean(
                torch.abs(
                    torch.atan2(pred_state_traj[:, :, 2], pred_state_traj[:, :, 3])
                    - torch.atan2(target_state_traj[:, :, 2], target_state_traj[:, :, 3])
                )
            )
        ).item()

        # get the indices of the samples in collision
        collision_idx = torch.any(target_collision_state_traj == 1, dim=1)
        # Offset in meters w.r.t. the target position relative to the traveled distance
        position_delta = torch.norm(pred_state_traj[:, -1, :2] - target_state_traj[:, -1, :2], dim=-1)
        distances = torch.norm(target_state_traj[:, -1, :2], dim=-1)
        rel_position_delta = position_delta / distances
        for distance in range(1, int(torch.max(distances).item()) + 1):
            samples_within_distance = torch.all(torch.vstack((distances - 1 < distance, distances > distance)), dim=0)
            # skip if no samples within distance
            if torch.sum(samples_within_distance) == 0:
                continue
            # mean of position offset
            meta[f"{mode}{suffix} Final Position Offset {distance-1} - {distance}m [Batch]"] = torch.mean(
                position_delta[samples_within_distance]
            ).item()
            # mean of position offset when in collision
            if torch.sum(collision_idx[samples_within_distance]) > 0:
                meta[f"{mode}{suffix} Final Position Offset {distance-1} - {distance}m [Collision]"] = torch.mean(
                    position_delta[collision_idx & samples_within_distance]
                ).item()
            if torch.sum(~collision_idx[samples_within_distance]) > 0:
                meta[f"{mode}{suffix} Final Position Offset {distance-1} - {distance}m [Non-Collision]"] = torch.mean(
                    position_delta[~collision_idx & samples_within_distance]
                ).item()
            # don't record for test set (overflow of information)
            if mode == "test":
                continue
            # std of position offset
            if torch.sum(samples_within_distance) > 1:
                meta[f"{mode}{suffix} Final Position Offset Std {distance-1} - {distance}m [Batch]"] = torch.std(
                    position_delta[samples_within_distance]
                ).item()
            # relative position offset
            meta[f"{mode}{suffix} Final Relative Position Offset {distance-1} - {distance}m [Batch]"] = torch.mean(
                rel_position_delta[samples_within_distance]
            ).item()

        # absolute position offset
        meta[f"{mode}{suffix} Final Position Offset [Batch]"] = torch.mean(position_delta).item()
        meta[f"{mode}{suffix} Final Position Offset [Batch] [Collision]"] = torch.mean(
            position_delta[collision_idx]
        ).item()
        meta[f"{mode}{suffix} Final Position Offset [Batch] [Non-Collision]"] = torch.mean(
            position_delta[~collision_idx]
        ).item()
        if mode != "test":
            meta[f"{mode}{suffix} Final Position Offset Std [Batch]"] = torch.std(position_delta).item()
            meta[f"{mode}{suffix} Final Position Offset Max [Batch]"] = torch.max(position_delta).item()
            meta[f"{mode}{suffix} Final Position Offset Min [Batch]"] = torch.min(position_delta).item()

        # relative position offset
        meta[f"{mode}{suffix} Final Relative Position Offset [Batch]"] = torch.mean(rel_position_delta).item()
        meta[f"{mode}{suffix} Final Relative Position Offset [Batch] [Collision]"] = torch.mean(
            rel_position_delta[collision_idx]
        ).item()
        meta[f"{mode}{suffix} Final Relative Position Offset [Batch] [Non-Collision]"] = torch.mean(
            rel_position_delta[~collision_idx]
        ).item()
        if mode != "test":
            meta[f"{mode}{suffix} Final Relative Position Offset Std [Batch]"] = torch.std(rel_position_delta).item()
            meta[f"{mode}{suffix} Final Relative Position Offset Max [Batch]"] = torch.max(rel_position_delta).item()
            meta[f"{mode}{suffix} Final Relative Position Offset Min [Batch]"] = torch.min(rel_position_delta).item()

        # change target depending on unified_predicition_value
        if self.cfg.unified_failure_prediction:
            target_collision_state_traj = torch.max(target_collision_state_traj, dim=-1)[0]
        # Precision of collision prediction
        meta[f"{mode}{suffix} Collision Prediction Precision [Batch]"] = self.metric_presision(
            pred_collision_prob_traj, target_collision_state_traj
        ).item()
        # Recall of collision prediction
        meta[f"{mode}{suffix} Collision Prediction Recall [Batch]"] = self.metric_recall(
            pred_collision_prob_traj, target_collision_state_traj
        ).item()
        # Accuracy of collision prediction
        meta[f"{mode}{suffix} Collision Prediction Accuracy [Batch]"] = self.metric_accuracy(
            pred_collision_prob_traj, target_collision_state_traj
        ).item()

        return meta


class FDMModelVelocityMultiStep(FDMModel):
    cfg: FDMBaseModelCfg
    """Model config class"""

    def __init__(self, cfg: FDMBaseModelCfg, device: str):
        super().__init__(cfg, device)

    """
    Forward function of the dynamics model
    """

    def forward(self, model_in: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        state, obs_proprioceptive, obs_extereoceptive, actions, add_obs_exteroceptive = (
            model_in[0].to(self.device),
            model_in[1].to(self.device),
            model_in[2].to(self.device),
            model_in[3].to(self.device),
            model_in[4].to(self.device),
        )

        # encode action trajectory if encoder is given
        batch_size, traj_len, single_action_dim = actions.shape

        ###
        # Normalize proprioceptive observations
        ###

        obs_proprioceptive = self.proprioceptive_normalizer(obs_proprioceptive)

        ###
        # Encode inputs
        ###

        # encode state and proprioceptive observations
        if isinstance(self.cfg.state_obs_proprioception_encoder, BaseModelCfg.MLPConfig):
            # MLP - only work with history equal to 1
            encoded_state_obs_proprioceptive = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
        else:
            # recurrent - work with history > 1
            # TODO: for GRU the same, for S4RNN check if the same as the hidden
            encoded_state_obs_proprioceptive, _ = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive[:, -1, :]

        # encode exteroceptive observations
        encoded_obs_exteroceptive = self.obs_exteroceptive_encoder(obs_extereoceptive)

        # concatenate last state and proprioceptive observation with exteroceptive observation
        if self.add_obs_exteroceptive_encoder is not None:
            encoded_add_obs_exteroceptive = self.add_obs_exteroceptive_encoder(add_obs_exteroceptive)
            encoded_state_obs = torch.concatenate(
                [encoded_state_obs_proprioceptive, encoded_obs_exteroceptive, encoded_add_obs_exteroceptive],
                dim=1,
            )
        else:
            encoded_state_obs = torch.concatenate([encoded_state_obs_proprioceptive, encoded_obs_exteroceptive], dim=1)

        # encode action trajectory if encoder is given
        if self.action_encoder is None:
            encoded_actions = actions
        else:
            encoded_actions = self.action_encoder.forward(actions.view(-1, single_action_dim)).view(
                batch_size, traj_len, -1
            )

        ###
        # Predict
        ###

        # predict friction
        friction = self.friction_predictor(encoded_state_obs_proprioceptive)
        encoded_state_obs = torch.concatenate([encoded_state_obs, friction], dim=1)

        # forward predict the state and observations
        if isinstance(self.cfg.recurrence, BaseModelCfg.MLPConfig):
            forward_predict_state_obs = self.recurrence(
                torch.concatenate([encoded_actions.flatten(start_dim=1), encoded_state_obs], dim=1)
            )
        else:
            # adjust the dimensions for the encoded_state_obs
            encoded_state_obs = encoded_state_obs.unsqueeze(1).repeat(1, traj_len, 1)

            # recurrent forward predict encoding of state and obsverations given the commands
            forward_predict_state_obs, _ = self.recurrence(
                torch.concatenate([encoded_actions, encoded_state_obs], dim=-1)
            )
            forward_predict_state_obs = forward_predict_state_obs.reshape(batch_size, -1)

        # predict the state transitions between consecutive commands in robot frame
        corr_vel = self.state_predictor.forward(forward_predict_state_obs)
        corr_vel = corr_vel.view(batch_size, traj_len, -1)

        # residual connection to velocity command
        corr_vel = corr_vel + actions

        # get the resulting change in position and angle when applying the commands perfectly
        # velocity command units x: [m/s], y: [m/s], phi: [rad/s]
        corr_distance = corr_vel * self.cfg.command_timestep

        # Cumsum is an inplace operation therefore the clone is necesasry
        cummulative_yaw = corr_distance.clone()[..., -1].cumsum(-1)

        # We need to take the non-linearity by the rotation into account
        r_vec1 = torch.stack([torch.cos(cummulative_yaw), -torch.sin(cummulative_yaw)], dim=-1)
        r_vec2 = torch.stack([torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=-1)
        so2 = torch.stack([r_vec1, r_vec2], dim=2)

        # Move the rotation in time and fill first timestep with identity - see math chapter
        so2 = torch.roll(so2, shifts=1, dims=1)
        so2[:, 0, :, :] = torch.eye(2, device=so2.device)[None].repeat(so2.shape[0], 1, 1)

        actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ corr_distance[..., :2].contiguous().reshape(-1, 2, 1)
        actions_local_frame = actions_local_frame.contiguous().reshape(so2.shape[0], so2.shape[1], 2)
        cumulative_position = (actions_local_frame).cumsum(-2)
        state_traj = torch.cat(
            [cumulative_position, torch.sin(cummulative_yaw)[:, :, None], torch.cos(cummulative_yaw)[:, :, None]],
            dim=-1,
        )

        # predict the probability of collision along the trajectory
        collision_prob_traj = self.sigmoid(self.collision_predictor.forward(forward_predict_state_obs))
        collision_prob_traj = collision_prob_traj.view(batch_size, traj_len)
        if self.cfg.unified_failure_prediction:
            collision_prob_traj = torch.max(collision_prob_traj, dim=-1)[0]

        # energy prediction
        energy_traj = self.energy_predictor(forward_predict_state_obs)
        energy_traj = energy_traj.view(batch_size, traj_len, -1)

        return state_traj, collision_prob_traj, energy_traj


class FDMModelVelocitySingleStep(FDMModel):
    cfg: FDMBaseModelCfg
    """Model config class"""

    def __init__(self, cfg: FDMBaseModelCfg, device: str):
        super().__init__(cfg, device)

    """
    Forward function of the dynamics model
    """

    def forward(self, model_in: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        state, obs_proprioceptive, obs_extereoceptive, actions, add_obs_exteroceptive = (
            model_in[0].to(self.device),
            model_in[1].to(self.device),
            model_in[2].to(self.device),
            model_in[3].to(self.device),
            model_in[4].to(self.device),
        )

        # encode action trajectory if encoder is given
        batch_size, traj_len, single_action_dim = actions.shape

        ###
        # Normalize proprioceptive observations
        ###

        obs_proprioceptive = self.proprioceptive_normalizer(obs_proprioceptive)

        ###
        # Encode inputs
        ###

        # encode state and proprioceptive observations
        if isinstance(self.cfg.state_obs_proprioception_encoder, BaseModelCfg.MLPConfig):
            # MLP - only work with history equal to 1
            encoded_state_obs_proprioceptive = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
        else:
            # recurrent - work with history > 1
            # TODO: for GRU the same, for S4RNN check if the same as the hidden
            encoded_state_obs_proprioceptive, _ = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive[:, -1, :]

        # encode exteroceptive observations
        encoded_obs_exteroceptive = self.obs_exteroceptive_encoder(obs_extereoceptive)

        # concatenate last state and proprioceptive observation with exteroceptive observation
        if self.add_obs_exteroceptive_encoder is not None:
            encoded_add_obs_exteroceptive = self.add_obs_exteroceptive_encoder(add_obs_exteroceptive)
            encoded_state_obs = torch.concatenate(
                [encoded_state_obs_proprioceptive, encoded_obs_exteroceptive, encoded_add_obs_exteroceptive],
                dim=1,
            )
        else:
            encoded_state_obs = torch.concatenate([encoded_state_obs_proprioceptive, encoded_obs_exteroceptive], dim=1)

        # encode action trajectory if encoder is given
        if self.action_encoder is None:
            encoded_actions = actions
        else:
            encoded_actions = self.action_encoder.forward(actions.view(-1, single_action_dim)).view(
                batch_size, traj_len, -1
            )

        ###
        # Predict
        ###

        # predict friction
        friction = self.friction_predictor(encoded_state_obs_proprioceptive)
        encoded_state_obs = torch.concatenate([encoded_state_obs, friction], dim=1)

        # adjust the dimensions for the encoded_state_obs
        encoded_state_obs = encoded_state_obs.unsqueeze(1)
        # initialize the hidden
        hidden = torch.zeros(
            self.cfg.recurrence.num_layers, batch_size, self.cfg.recurrence.hidden_size, device=self.device
        )
        # init buffers for corr_vel, collision and engergy predictions
        corr_vel = torch.zeros(batch_size, traj_len, 3, device=self.device)
        collision_prob_traj = torch.zeros(batch_size, traj_len, device=self.device)
        energy_traj = torch.zeros(batch_size, traj_len, device=self.device)
        last_corr_vel = torch.zeros(batch_size, 1, 3, device=self.device)
        last_collision_prob = torch.zeros(batch_size, 1, 1, device=self.device)
        last_energy = torch.zeros(batch_size, 1, 1, device=self.device)

        for traj_idx in range(traj_len):
            # recurrent forward predict encoding of state and obsverations given the commands
            forward_predict_state_obs, hidden = self.recurrence(
                torch.concatenate(
                    [
                        encoded_actions[:, traj_idx].unsqueeze(1),
                        encoded_state_obs,
                        last_corr_vel,
                        last_collision_prob,
                        last_energy,
                    ],
                    dim=-1,
                ),
                hidden,
            )
            forward_predict_state_obs = forward_predict_state_obs.squeeze(1)

            # predict the state transitions between consecutive commands in robot frame
            corr_vel[:, traj_idx] = self.state_predictor.forward(forward_predict_state_obs)
            last_corr_vel = corr_vel[:, traj_idx].unsqueeze(1)

            # predict the probability of collision along the trajectory
            collision_prob_traj[:, traj_idx] = self.sigmoid(
                self.collision_predictor.forward(forward_predict_state_obs)
            ).squeeze(-1)
            last_collision_prob = collision_prob_traj[:, traj_idx].unsqueeze(1).unsqueeze(-1)

            # energy prediction
            energy_traj[:, traj_idx] = self.energy_predictor(forward_predict_state_obs).squeeze(-1)
            last_energy = energy_traj[:, traj_idx].unsqueeze(1).unsqueeze(-1)

        # residual connection to velocity command
        corr_vel = corr_vel + actions

        # get the resulting change in position and angle when applying the commands perfectly
        # velocity command units x: [m/s], y: [m/s], phi: [rad/s]
        corr_distance = corr_vel * self.cfg.command_timestep

        # Cumsum is an inplace operation therefore the clone is necesasry
        cummulative_yaw = corr_distance.clone()[..., -1].cumsum(-1)

        # We need to take the non-linearity by the rotation into account
        r_vec1 = torch.stack([torch.cos(cummulative_yaw), -torch.sin(cummulative_yaw)], dim=-1)
        r_vec2 = torch.stack([torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=-1)
        so2 = torch.stack([r_vec1, r_vec2], dim=2)

        # Move the rotation in time and fill first timestep with identity - see math chapter
        so2 = torch.roll(so2, shifts=1, dims=1)
        so2[:, 0, :, :] = torch.eye(2, device=so2.device)[None].repeat(so2.shape[0], 1, 1)

        actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ corr_distance[..., :2].contiguous().reshape(-1, 2, 1)
        actions_local_frame = actions_local_frame.contiguous().reshape(so2.shape[0], so2.shape[1], 2)
        cumulative_position = (actions_local_frame).cumsum(-2)
        state_traj = torch.cat(
            [cumulative_position, torch.sin(cummulative_yaw)[:, :, None], torch.cos(cummulative_yaw)[:, :, None]],
            dim=-1,
        )

        # predict the probability of collision along the trajectory if unified prediction
        if self.cfg.unified_failure_prediction:
            collision_prob_traj = torch.max(collision_prob_traj, dim=-1)[0]

        return state_traj, collision_prob_traj, energy_traj


class FDMModelVelocitySingleStepHeightAdjust(FDMModel):
    cfg: FDMModelVelocitySingleStepHeightAdjustCfg
    """Model config class"""

    def __init__(self, cfg: FDMModelVelocitySingleStepHeightAdjustCfg, device: str):
        super().__init__(cfg, device)

        # Create a meshgrid of coordinates
        self.idx_tensor_x, self.idx_tensor_y = torch.meshgrid(
            torch.arange(
                self.cfg.scan_cut_dim_y[0] / self.cfg.height_scan_res,
                (self.cfg.scan_cut_dim_y[1] / self.cfg.height_scan_res) + 1,
                device=device,
            ),
            torch.arange(
                self.cfg.scan_cut_dim_x[0] / self.cfg.height_scan_res,
                (self.cfg.scan_cut_dim_x[1] / self.cfg.height_scan_res) + 1,
                device=device,
            ),
            indexing="ij",
        )
        self.idx_tensor_x = self.idx_tensor_x.flatten().float()
        self.idx_tensor_y = self.idx_tensor_y.flatten().float()

    """
    Forward function of the dynamics model
    """

    def forward(self, model_in: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        state, obs_proprioceptive, obs_extereoceptive, actions, add_obs_exteroceptive = (
            model_in[0].to(self.device),
            model_in[1].to(self.device),
            model_in[2].to(self.device),
            model_in[3].to(self.device),
            model_in[4].to(self.device),
        )

        # encode action trajectory if encoder is given
        batch_size, traj_len, single_action_dim = actions.shape

        ###
        # Normalize proprioceptive observations
        ###
        obs_proprioceptive = self.proprioceptive_normalizer(obs_proprioceptive)

        ###
        # Encode inputs
        ###

        # encode state and proprioceptive observations
        if isinstance(self.cfg.state_obs_proprioception_encoder, BaseModelCfg.MLPConfig):
            # MLP - only work with history equal to 1
            encoded_state_obs_proprioceptive = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
        else:
            # recurrent - work with history > 1
            # TODO: for GRU the same, for S4RNN check if the same as the hidden
            encoded_state_obs_proprioceptive, _ = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive[:, -1, :]

        # concatenate last state and proprioceptive observation with extra exteroceptive observation
        if self.add_obs_exteroceptive_encoder is not None:
            encoded_add_obs_exteroceptive = self.add_obs_exteroceptive_encoder(add_obs_exteroceptive)
            encoded_state_obs_proprioceptive = torch.concatenate(
                [encoded_state_obs_proprioceptive, encoded_add_obs_exteroceptive],
                dim=1,
            )

        # encode action trajectory if encoder is given
        if self.action_encoder is None:
            encoded_actions = actions
        else:
            encoded_actions = self.action_encoder.forward(actions.view(-1, single_action_dim)).view(
                batch_size, traj_len, -1
            )

        ###
        # Predict
        ###

        # predict friction
        friction = self.friction_predictor(encoded_state_obs_proprioceptive)
        encoded_state_obs = torch.concatenate([encoded_state_obs_proprioceptive, friction], dim=1)

        # adjust the dimensions for the encoded_state_obs
        encoded_state_obs = encoded_state_obs.unsqueeze(1)
        # initialize the hidden
        hidden = torch.zeros(
            self.cfg.recurrence.num_layers, batch_size, self.cfg.recurrence.hidden_size, device=self.device
        )
        # init buffers for collision and engergy predictions
        collision_prob_traj = torch.zeros(batch_size, traj_len, device=self.device)
        energy_traj = torch.zeros(batch_size, traj_len, device=self.device)
        last_corr_vel = torch.zeros(batch_size, 3, device=self.device)
        last_collision_prob = torch.zeros(batch_size, 1, 1, device=self.device)
        last_energy = torch.zeros(batch_size, 1, 1, device=self.device)
        # buffer for the height scan crop
        curr_state = state[:, 0, :4]
        cummulative_yaw = torch.zeros(batch_size, device=self.device)
        state_traj = torch.zeros(batch_size, traj_len, 4, device=self.device)

        for traj_idx in range(traj_len):
            # cut height scan
            obs_extereoceptive_cut = self.height_scan_cut(curr_state, obs_extereoceptive, traj_idx=traj_idx)

            # encode exteroceptive observations
            encoded_obs_exteroceptive = self.obs_exteroceptive_encoder(obs_extereoceptive_cut)

            # recurrent forward predict encoding of state and obsverations given the commands
            forward_predict_state_obs, hidden = self.recurrence(
                torch.concatenate(
                    [
                        encoded_actions[:, traj_idx].unsqueeze(1),
                        encoded_state_obs,
                        encoded_obs_exteroceptive.unsqueeze(1),
                        last_corr_vel.unsqueeze(1),
                        last_collision_prob,
                        last_energy,
                    ],
                    dim=-1,
                ),
                hidden,
            )
            forward_predict_state_obs = forward_predict_state_obs.squeeze(1)

            # predict the state transitions between consecutive commands in robot frame
            last_corr_vel = self.state_predictor.forward(forward_predict_state_obs)

            # predict the probability of collision along the trajectory
            collision_prob_traj[:, traj_idx] = self.sigmoid(
                self.collision_predictor.forward(forward_predict_state_obs)
            ).squeeze(-1)
            last_collision_prob = collision_prob_traj[:, traj_idx].unsqueeze(1).unsqueeze(-1)

            # energy prediction
            energy_traj[:, traj_idx] = self.energy_predictor(forward_predict_state_obs).squeeze(-1)
            last_energy = energy_traj[:, traj_idx].unsqueeze(1).unsqueeze(-1)

            # residual connection to velocity command
            # get the resulting change in position and angle when applying the commands perfectly
            # velocity command units x: [m/s], y: [m/s], phi: [rad/s]
            corr_distance = (last_corr_vel + actions[:, traj_idx]) * self.cfg.command_timestep

            cummulative_yaw = cummulative_yaw + corr_distance[..., -1]

            # We need to take the non-linearity by the rotation into account
            r_vec1 = torch.stack([torch.cos(cummulative_yaw), -torch.sin(cummulative_yaw)], dim=-1)
            r_vec2 = torch.stack([torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=-1)
            so2 = torch.stack([r_vec1, r_vec2], dim=2)

            # TODO: check if this is correct

            actions_local_frame = (so2.contiguous() @ corr_distance[..., :2].contiguous().reshape(-1, 2, 1)).squeeze(-1)
            state_traj[:, traj_idx, :2] = curr_state[:, :2] + actions_local_frame
            state_traj[:, traj_idx, 2] = torch.sin(cummulative_yaw)
            state_traj[:, traj_idx, 3] = torch.cos(cummulative_yaw)
            # save current state for next iteration
            curr_state = state_traj[:, traj_idx]

        # predict the probability of collision along the trajectory if unified prediction
        if self.cfg.unified_failure_prediction:
            collision_prob_traj = torch.max(collision_prob_traj, dim=-1)[0]

        return state_traj, collision_prob_traj, energy_traj

    def height_scan_cut(self, curr_state, obs_extereoceptive: torch.Tensor, traj_idx: int) -> torch.Tensor:

        # get effective translation
        # since in robot frame, the y translation is against the height axis x direction, has to be negative
        effective_translation_tensor_x = (
            -curr_state[:, 1] / self.cfg.height_scan_res + self.cfg.height_scan_robot_center[0]
        )
        effective_translation_tensor_y = (
            curr_state[:, 0] / self.cfg.height_scan_res + self.cfg.height_scan_robot_center[1]
        )

        idx_tensor_x = self.idx_tensor_x.repeat(curr_state.shape[0], 1)
        idx_tensor_y = self.idx_tensor_y.repeat(curr_state.shape[0], 1)

        # angle definition for the height scan coordinate system is opposite of the tensor system, so negative
        s = curr_state[:, 2].unsqueeze(1)
        c = curr_state[:, 3].unsqueeze(1)
        idx_crop_x = (c * idx_tensor_x - s * idx_tensor_y + effective_translation_tensor_x.unsqueeze(1)).round().int()
        idx_crop_y = (s * idx_tensor_x + c * idx_tensor_y + effective_translation_tensor_y.unsqueeze(1)).round().int()

        # move idx tensors of the new image to 0,0 in upper left corner
        idx_tensor_x += torch.abs(torch.min(idx_tensor_x, dim=-1)[0]).unsqueeze(1)
        idx_tensor_y += torch.abs(torch.min(idx_tensor_y, dim=-1)[0]).unsqueeze(1)
        idx_tensor_x = idx_tensor_x.round().int()
        idx_tensor_y = idx_tensor_y.round().int()

        # filter_idx outside the image
        filter_idx = (
            (idx_crop_x >= 0)
            & (idx_crop_x < self.cfg.height_scan_shape[0])
            & (idx_crop_y >= 0)
            & (idx_crop_y < self.cfg.height_scan_shape[1])
        )
        idx_crop_x[~filter_idx] = 0
        idx_crop_y[~filter_idx] = 0

        new_image = torch.zeros(
            (
                curr_state.shape[0],
                math.ceil((self.cfg.scan_cut_dim_y[1] - self.cfg.scan_cut_dim_y[0]) / self.cfg.height_scan_res + 1),
                math.ceil((self.cfg.scan_cut_dim_x[1] - self.cfg.scan_cut_dim_x[0]) / self.cfg.height_scan_res + 1),
            ),
            device=self.device,
        )
        ALL_INDICES = (
            torch.arange(curr_state.shape[0], device=self.device)
            .int()[:, None]
            .repeat(1, new_image.shape[1] * new_image.shape[2])
        )
        obs_extereoceptive = obs_extereoceptive.squeeze(1)
        new_image[ALL_INDICES, idx_tensor_x, idx_tensor_y] = obs_extereoceptive[ALL_INDICES, idx_crop_x, idx_crop_y]

        filter_idx_nonzero = (~filter_idx).nonzero()
        new_image[
            filter_idx_nonzero[:, 0].int(),
            idx_tensor_x[filter_idx_nonzero[:, 0], filter_idx_nonzero[:, 1]],
            idx_tensor_y[filter_idx_nonzero[:, 0], filter_idx_nonzero[:, 1]],
        ] = 0

        if False:
            import matplotlib.pyplot as plt

            # Visualization using matplotlib
            idx = 0
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            vmin = torch.min(obs_extereoceptive[idx]).item()
            vmax = 1

            img = axs[0].imshow(obs_extereoceptive[idx].cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            axs[0].set_title("Large Height Scan")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")

            axs[1].imshow(new_image[idx].cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            axs[1].set_title(
                f"{curr_state[idx, 0].float():.4f} {curr_state[idx, 1].float():.4f} {torch.atan2(curr_state[idx, 2], curr_state[idx, 3]).float():.4f}"
            )
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")

            mask = torch.zeros(*self.cfg.height_scan_shape, dtype=torch.bool, device=self.device)
            mask[idx_crop_x[idx], idx_crop_y[idx]] = True
            masked_image = torch.where(mask, obs_extereoceptive[idx], torch.tensor(1))

            axs[2].imshow(masked_image.cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            axs[2].set_xlabel("X")
            axs[2].set_ylabel("Y")

            # Create a colorbar
            cbar = fig.colorbar(img, ax=axs, fraction=0.02, pad=0.04)
            cbar.set_label("Color Scale")

            plt.tight_layout()
            plt.savefig(f"height_scan_network_{traj_idx}.png")
            plt.close()

        # cut height scan
        return new_image.unsqueeze(1)
