

from __future__ import annotations

import prettytable
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from .model_base import Model
from .model_base_cfg import BaseModelCfg
from .utils import EmpiricalNormalization, L2Loss

if TYPE_CHECKING:
    from .fdm_proprioception_model_cfg import FDMProprioceptionModelCfg, FDMProprioceptionVelocityModelCfg


class FDMProprioceptionModel(Model):
    cfg: FDMProprioceptionModelCfg
    """Model config class"""

    def __init__(self, cfg: FDMProprioceptionModelCfg, device: str):
        super().__init__(cfg, device)

        # build layers
        self.state_obs_proprioceptive_encoder = self._construct_layer(self.cfg.state_obs_proprioception_encoder)
        self.recurrence = self._construct_layer(self.cfg.recurrence)
        self.state_predictor = self._construct_layer(self.cfg.state_predictor)
        self.friction_predictor = self._construct_layer(self.cfg.friction_predictor)
        if self.cfg.action_encoder is not None:
            self.action_encoder = self._construct_layer(self.cfg.action_encoder)
        else:
            self.action_encoder = None

        # init loss functions
        if self.cfg.pos_loss_norm == "mse":
            self.position_loss = nn.MSELoss()
        elif self.cfg.pos_loss_norm == "l2":
            self.position_loss = L2Loss
        elif self.cfg.pos_loss_norm == "l1":
            self.position_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown position loss norm: {self.cfg.pos_loss_norm}")

        self.heading_loss = nn.MSELoss()
        self.perfect_velocity_position_loss = nn.MSELoss()
        self.stop_loss = nn.MSELoss()
        self.friction_loss = nn.MSELoss()

        # init velocity and acceleration limit buffer --> filled by maximum oberserved simulation values
        self.acceleration_limits = torch.zeros(3, device=self.device)
        self.velocity_limits = torch.zeros(3, device=self.device)

        # include empirical normalizer for the proprioceptive observations
        self.proprioceptive_normalizer = EmpiricalNormalization(self.cfg.empirical_normalization_dim)

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
    Forward function of the dynamics model
    """

    def forward(self, model_in: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
        state, obs_proprioceptive, actions = (
            model_in[0].to(self.device),
            model_in[1].to(self.device),
            model_in[3].to(self.device),
        )

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
            encoded_state_obs_proprioceptive, _ = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive[:, -1, :]

        # encode action trajectory if encoder is given
        batch_size, traj_len, single_action_dim = actions.shape
        if self.action_encoder is None:
            encoded_actions = actions
        else:
            actions = actions.view(-1, single_action_dim) * self.cfg.command_timestep
            encoded_actions = self.action_encoder.forward(actions).view(batch_size, traj_len, -1)

        ###
        # Predict
        ###

        # predict friction
        friction = self.friction_predictor(encoded_state_obs_proprioceptive)

        if isinstance(self.cfg.recurrence, BaseModelCfg.MLPConfig):
            forward_predict_state_obs = self.recurrence(
                torch.concatenate([encoded_actions.flatten(start_dim=1), encoded_state_obs_proprioceptive], dim=1)
            )
        else:
            # adjust the dimensions for the encoded_state_obs_proprioceptive
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive.unsqueeze(1).repeat(1, traj_len, 1)

            # recurrent forward predict encoding of state and obsverations given the commands
            forward_predict_state_obs, _ = self.recurrence(
                torch.concatenate([encoded_actions, encoded_state_obs_proprioceptive], dim=-1)
            )
            forward_predict_state_obs = forward_predict_state_obs.reshape(batch_size, -1)

        # predict the state transitions between consecutive commands in robot frame
        rel_state_transitions = self.state_predictor.forward(forward_predict_state_obs)
        rel_state_transitions = rel_state_transitions.view(batch_size, traj_len, -1)

        # add relative transitions to previous ones to get absolute transitions
        state_traj = torch.cumsum(rel_state_transitions, dim=1)

        # if isinstance(self.cfg.recurrence, BaseModelCfg.MLPConfig):
        #     forward_predict_state_obs = self.recurrence(
        #         torch.concatenate([encoded_actions.flatten(start_dim=1), encoded_state_obs_proprioceptive], dim=1)
        #     )
        # else:
        #     hidden = torch.zeros((self.cfg.recurrence.num_layers, batch_size, self.cfg.recurrence.hidden_size), device=self.device)
        #     state_traj = torch.zeros((batch_size, traj_len, self.cfg.state_predictor.output), device=self.device)
        #     prev_state_transitions = torch.zeros((batch_size, self.cfg.state_predictor.output), device=self.device)
        #     prev_rel_state_transitions = torch.zeros((batch_size, self.cfg.state_predictor.output), device=self.device)

        #     for idx in range(traj_len):
        #         forward_predict_state_obs, hidden = self.recurrence(
        #             torch.concatenate([encoded_actions[:, idx], encoded_state_obs_proprioceptive, prev_state_transitions, prev_rel_state_transitions], dim=1).unsqueeze(1), hidden
        #         )
        #         # prev_rel_state_transitions = self.state_predictor(forward_predict_state_obs.squeeze(1))

        #         # # transform relative state transitions to base frame and add all previous state transitions
        #         # state_traj[:, idx, 2:] = prev_rel_state_transitions[:, 2:] + prev_state_transitions[:, 2:]
        #         # so2 = torch.stack([state_traj[:, idx, [2, 3]] * torch.tensor([1, -1], device=self.device), state_traj[:, idx, [3, 2]]], dim=2)
        #         # state_traj[:, idx, :2] = (so2 @ prev_rel_state_transitions[:, :2, None]).squeeze(-1) + prev_state_transitions[:, :2]

        #         # # update previous state transitions
        #         # prev_state_transitions = state_traj[:, idx]

        #         state_traj[:, idx] = self.state_predictor(forward_predict_state_obs.squeeze(1))
        #         prev_rel_state_transitions = state_traj[:, idx].clone()
        #         prev_state_transitions = torch.sum(state_traj, dim=1)

        # # TODO: implement as correction for perfect velocity model
        # # add relative transitions to previous ones to get absolute transitions
        # state_traj = torch.cumsum(state_traj, dim=1)

        return state_traj, friction

    def loss(
        self,
        model_out: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mode: str = "train",
        suffix: str = "",
    ) -> tuple[torch.Tensor, dict]:
        """Network loss function as a combintation of the collision probability loss and the coordinate loss"""
        # extract predictions and targets
        pred_state_traj, pred_friction = model_out[0], model_out[1]
        target_state_traj, target_collision_state_traj, target_friction = (
            target[..., :4].to(self.device),
            target[..., 4].to(self.device),
            target[..., -1, -4:].to(self.device),
        )

        # stop loss - do not move when collision has happenend
        stop_loss = self.stop_loss(
            pred_state_traj[target_collision_state_traj == 1], target_state_traj[target_collision_state_traj == 1]
        )
        stop_loss = torch.tensor(0.0, device=self.device) if torch.isnan(stop_loss) else stop_loss

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

        # get change in position and heading
        position_delta = torch.cat(
            [pred_state_traj[:, 0, :2].unsqueeze(1), pred_state_traj[:, 1:, :2] - pred_state_traj[:, :-1, :2]], dim=1
        )
        # convert heading representation to radians
        pred_heading_change = torch.atan2(pred_state_traj[:, :, 2], pred_state_traj[:, :, 3])
        heading_delta = torch.cat(
            [pred_heading_change[:, 0].unsqueeze(1), pred_heading_change[:, 1:] - pred_heading_change[:, :-1]], dim=1
        )
        # enforce periodicity of the heading
        heading_delta = torch.abs(heading_delta)
        heading_delta = torch.minimum((2 * torch.pi) - heading_delta, heading_delta)
        # combine position and heading delta
        pose_delta = torch.cat([position_delta, heading_delta.unsqueeze(2)], dim=2)

        velocity = pose_delta / self.cfg.command_timestep
        # Velocity loss (sum of violations)
        velocity_loss = torch.abs(velocity) - self.velocity_limits
        velocity_loss = torch.sum(velocity_loss.clip(min=0.0))

        # get acceleration
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / self.cfg.command_timestep
        # Acceleration loss (sum of violations)
        acceleration_loss = torch.abs(acceleration) - self.acceleration_limits
        acceleration_loss = torch.sum(acceleration_loss.clip(min=0.0))

        # friction loss
        friction_loss = self.friction_loss(pred_friction, target_friction)

        # combine losses
        loss = (
            position_loss * self.cfg.loss_weights["position"]
            + heading_loss * self.cfg.loss_weights["heading"]
            + velocity_loss * self.cfg.loss_weights["velocity"]
            + acceleration_loss * self.cfg.loss_weights["acceleration"]
            + stop_loss * self.cfg.loss_weights["stop"]
            + friction_loss * self.cfg.loss_weights["friction"]
        )

        # save meta data
        meta = {
            f"{mode}{suffix} Loss [Batch]": loss.item(),
            f"{mode}{suffix} Position Loss [Batch]": position_loss.item(),
            f"{mode}{suffix} Heading Loss [Batch]": heading_loss.item(),
            f"{mode}{suffix} Velocity Loss [Batch]": velocity_loss.item(),
            f"{mode}{suffix} Acceleration Loss [Batch]": acceleration_loss.item(),
            f"{mode}{suffix} Stop Loss [Batch]": stop_loss.item(),
            f"{mode}{suffix} Friction Loss [Batch]": friction_loss.item(),
        }
        [
            meta.update({f"{mode}{suffix} Position Loss Horizon {idx} [Batch]": position_loss_list[idx].item()})
            for idx in range(0, len(position_loss_list), int(len(position_loss_list) / 5))
        ]
        meta.update({f"{mode}{suffix} Position Loss Horizon Last [Batch]": position_loss_list[-1].item()})
        [
            meta.update({f"{mode}{suffix} Heading Loss Horizon {idx} [Batch]": heading_loss_list[idx].item()})
            for idx in range(0, len(heading_loss_list), int(len(heading_loss_list) / 5))
        ]
        meta.update({f"{mode}{suffix} Heading Loss Horizon Last [Batch]": heading_loss_list[-1].item()})

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
        pred_state_traj = model_out[0]
        target_state_traj = target[..., :4].to(self.device)

        if meta is None:
            meta = {}

        # compare to perfect velocity tracking error when eval mode
        if mode == "eval" and "eval Position Loss [Batch]" in meta:
            if isinstance(eval_in, (tuple, list)):
                eval_in = eval_in[0].to(self.device)
            else:
                eval_in = eval_in.to(self.device)
            perf_vel_loss = self.perfect_velocity_position_loss(eval_in, target_state_traj).item()
            pred_loss = self.perfect_velocity_position_loss(pred_state_traj, target_state_traj).item()
            meta[f"{mode}{suffix} Pred Loss / Perfect Velocity Loss [Batch]"] = pred_loss / perf_vel_loss

        # Heading error in degrees
        yaw_diff = torch.abs(
            torch.atan2(pred_state_traj[:, :, 2], pred_state_traj[:, :, 3])
            - torch.atan2(target_state_traj[:, :, 2], target_state_traj[:, :, 3])
        )
        # enforce periodicity of the heading
        yaw_diff = torch.minimum((2 * torch.pi) - yaw_diff, yaw_diff)
        meta[f"{mode}{suffix} Heading Degree Error [Batch]"] = torch.rad2deg(torch.mean(yaw_diff)).item()

        # Offset in meters w.r.t. the target position relative to the traveled distance
        position_delta = torch.norm(pred_state_traj[:, -1, :2] - target_state_traj[:, -1, :2], dim=-1, p=2)
        distances = torch.norm(target_state_traj[:, -1, :2], dim=-1)
        rel_position_delta = position_delta / distances
        for distance in range(1, int(torch.max(distances).item()) + 1):
            samples_within_distance = torch.all(torch.vstack((distances - 1 < distance, distances > distance)), dim=0)
            # skip if no samples within distance
            if torch.sum(samples_within_distance) == 0:
                continue
            meta[f"{mode}{suffix} Final Position Offset {distance-1} - {distance}m [Batch]"] = torch.mean(
                position_delta[samples_within_distance]
            ).item()
            if torch.sum(samples_within_distance) > 1:
                meta[f"{mode}{suffix} Final Position Offset Std {distance-1} - {distance}m [Batch]"] = torch.std(
                    position_delta[samples_within_distance]
                ).item()
            meta[f"{mode}{suffix} Final Relative Position Offset {distance-1} - {distance}m [Batch]"] = torch.mean(
                rel_position_delta[samples_within_distance]
            ).item()
        meta[f"{mode}{suffix} Final Position Offset [Batch]"] = torch.mean(position_delta).item()
        meta[f"{mode}{suffix} Final Relative Position Offset [Batch]"] = torch.mean(rel_position_delta).item()

        return meta


class FDMProprioceptionVelocityModel(FDMProprioceptionModel):
    cfg: FDMProprioceptionVelocityModelCfg
    """Model config class"""

    def __init__(self, cfg: FDMProprioceptionVelocityModelCfg, device: str):
        super().__init__(cfg, device)

    def forward(self, model_in: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
        state, obs_proprioceptive, actions = (
            model_in[0].to(self.device),
            model_in[1].to(self.device),
            model_in[3].to(self.device),
        )

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
            encoded_state_obs_proprioceptive, _ = self.state_obs_proprioceptive_encoder(
                torch.concatenate([state, obs_proprioceptive], dim=2)
            )
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive[:, -1, :]

        # encode action trajectory if encoder is given
        batch_size, traj_len, single_action_dim = actions.shape
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

        if isinstance(self.cfg.recurrence, BaseModelCfg.MLPConfig):
            forward_predict_state_obs = self.recurrence(
                torch.concatenate([encoded_actions.flatten(start_dim=1), encoded_state_obs_proprioceptive], dim=1)
            )
        else:
            # adjust the dimensions for the encoded_state_obs_proprioceptive
            encoded_state_obs_proprioceptive = encoded_state_obs_proprioceptive.unsqueeze(1).repeat(1, traj_len, 1)

            # recurrent forward predict encoding of state and obsverations given the commands
            forward_predict_state_obs, _ = self.recurrence(
                torch.concatenate([encoded_actions, encoded_state_obs_proprioceptive], dim=-1)
            )
            forward_predict_state_obs = forward_predict_state_obs.reshape(batch_size, -1)

        # predict the state transitions between consecutive commands in robot frame
        corr_vel = self.state_predictor.forward(forward_predict_state_obs)
        corr_vel = corr_vel.view(batch_size, traj_len, -1)

        # residual connection to velocity command
        # TODO: check if that improves the performance
        corr_vel = corr_vel + actions

        # if isinstance(self.cfg.recurrence, BaseModelCfg.MLPConfig):
        #     forward_predict_state_obs = self.recurrence(
        #         torch.concatenate([encoded_actions.flatten(start_dim=1), encoded_state_obs_proprioceptive], dim=1)
        #     )
        # else:
        #     hidden = torch.zeros((self.cfg.recurrence.num_layers, batch_size, self.cfg.recurrence.hidden_size), device=self.device)
        #     state_traj = torch.zeros((batch_size, traj_len, self.cfg.state_predictor.output), device=self.device)
        #     prev_state_transitions = torch.zeros((batch_size, self.cfg.state_predictor.output), device=self.device)
        #     prev_rel_state_transitions = torch.zeros((batch_size, self.cfg.state_predictor.output), device=self.device)

        #     for idx in range(traj_len):
        #         forward_predict_state_obs, hidden = self.recurrence(
        #             torch.concatenate([encoded_actions[:, idx], encoded_state_obs_proprioceptive, prev_state_transitions, prev_rel_state_transitions], dim=1).unsqueeze(1), hidden
        #         )
        #         # prev_rel_state_transitions = self.state_predictor(forward_predict_state_obs.squeeze(1))

        #         # # transform relative state transitions to base frame and add all previous state transitions
        #         # state_traj[:, idx, 2:] = prev_rel_state_transitions[:, 2:] + prev_state_transitions[:, 2:]
        #         # so2 = torch.stack([state_traj[:, idx, [2, 3]] * torch.tensor([1, -1], device=self.device), state_traj[:, idx, [3, 2]]], dim=2)
        #         # state_traj[:, idx, :2] = (so2 @ prev_rel_state_transitions[:, :2, None]).squeeze(-1) + prev_state_transitions[:, :2]

        #         # # update previous state transitions
        #         # prev_state_transitions = state_traj[:, idx]

        #         state_traj[:, idx] = self.state_predictor(forward_predict_state_obs.squeeze(1))
        #         prev_rel_state_transitions = state_traj[:, idx].clone()
        #         prev_state_transitions = torch.sum(state_traj, dim=1)

        # # TODO: implement as correction for perfect velocity model
        # # add relative transitions to previous ones to get absolute transitions
        # state_traj = torch.cumsum(state_traj, dim=1)

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

        return state_traj, friction
