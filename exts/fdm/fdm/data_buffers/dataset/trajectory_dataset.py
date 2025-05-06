# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset
from typing import TYPE_CHECKING

import pypose as pp

import isaaclab.utils.math as math_utils

from fdm import VEL_RANGE_X, VEL_RANGE_Y, VEL_RANGE_YAW

if TYPE_CHECKING:
    from fdm.data_buffers import ReplayBuffer, ReplayBufferCfg
    from fdm.model import FDMBaseModelCfg
    from fdm.runner import TrainerBaseCfg


class TrajectoryDataset(Dataset):
    def __init__(
        self, cfg: TrainerBaseCfg, model_cfg: FDMBaseModelCfg, replay_buffer_cfg: ReplayBufferCfg, return_device: str
    ):
        # save configs
        self.cfg: TrainerBaseCfg = cfg
        self.model_cfg: FDMBaseModelCfg = model_cfg
        self.replay_buffer_cfg: ReplayBufferCfg = replay_buffer_cfg
        self._actual_nbr_samples: int = self.cfg.num_samples
        self.return_device: str = return_device

        # save min and max of the hard_contact_obs (as part of the state) for normalization
        self.min_hard_contact_obs = torch.tensor([torch.inf])
        self.max_hard_contact_obs = torch.zeros(1)

        # init extereoceptive noise model
        if self.cfg.extereoceptive_noise_model is not None:
            self.extereoceptive_noise_model = self.cfg.extereoceptive_noise_model.noise_model(
                self.cfg.extereoceptive_noise_model, device=self.replay_buffer_cfg.buffer_device
            )
        else:
            self.extereoceptive_noise_model = None

    def __str__(self) -> str:
        msg = (
            "#############################################################################################\n"
            f"<RandomTrajectoryDataset> with command trajectory (length {self.replay_buffer_cfg.trajectory_length})"
            " contains\n"
            f"\tIntended Number: \t{self.cfg.num_samples}\n"
            f"\tCollision rate : \t{self.collision_rate})\n"
            f"\tReturn Device  : \t{self.return_device}\n"
            "#############################################################################################"
        )

        return msg

    ##
    # Properties
    ##

    @property
    def collision_sample_nb(self) -> int:
        return torch.any(self.states[..., 4], axis=1).sum()

    @property
    def collision_rate(self) -> float:
        return self.collision_sample_nb / self.__len__()

    @property
    def num_samples(self) -> int:
        return self._actual_nbr_samples

    ##
    # Operations
    ##

    def populate(  # noqa: C901
        self, replay_buffer: ReplayBuffer, regular_slicing: bool = False, start_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update data in the buffer for specified indexes.

        Args:
            replay_buffer: The replay buffer to get the data from.
            regular_slicing: If True, the data is sliced regularly.
            start_idx: The start indexes for the different environments where the samples should be taken from.
        """
        # get start and end indexes for the different environments
        if regular_slicing and start_idx is None:
            trajectory_idx = torch.arange(
                0,
                self.replay_buffer_cfg.trajectory_length - self.model_cfg.prediction_horizon - 1,
                device=self.replay_buffer_cfg.buffer_device,
            )[1 :: self.model_cfg.prediction_horizon]
            start_idx = torch.vstack((
                torch.arange(0, replay_buffer.env.num_envs, device=self.replay_buffer_cfg.buffer_device)[:, None]
                .repeat(1, len(trajectory_idx))
                .flatten(),
                trajectory_idx.repeat(replay_buffer.env.num_envs),
            )).T
        elif start_idx is None:
            traj_start_idx = self._sample_random_traj_idx(replay_buffer)
            coll_start_idx = self._sample_collision_traj(replay_buffer)

            # balance the data
            if self.cfg.collision_rate is not None:
                assert (
                    int(self.cfg.num_samples * (1 - self.cfg.collision_rate)) <= traj_start_idx.shape[0]
                ), "Not enough regular samples to balance data!"
                perm = torch.randperm(traj_start_idx.shape[0], device=self.replay_buffer_cfg.buffer_device)
                traj_start_idx = traj_start_idx[perm[: int(self.cfg.num_samples * (1 - self.cfg.collision_rate))]]
                if int(self.cfg.num_samples * self.cfg.collision_rate) <= coll_start_idx.shape[0]:
                    perm = torch.randperm(coll_start_idx.shape[0], device=self.replay_buffer_cfg.buffer_device)
                    coll_start_idx = coll_start_idx[perm[: int(self.cfg.num_samples * self.cfg.collision_rate)]]
                else:
                    coll_start_idx = coll_start_idx.repeat(
                        int(self.cfg.num_samples * self.cfg.collision_rate) // coll_start_idx.shape[0] + 1, 1
                    )
                    coll_start_idx = coll_start_idx[: int(self.cfg.num_samples * self.cfg.collision_rate)]
                start_idx = torch.vstack([traj_start_idx, coll_start_idx])
            else:
                start_idx = torch.vstack([traj_start_idx, coll_start_idx])

        ###
        # Actions
        ###
        self.actions = torch.concatenate(
            [
                replay_buffer.actions[start_idx[:, 0], start_idx[:, 1] + idx][:, None, :]
                for idx in range(self.model_cfg.prediction_horizon)
            ],
            dim=1,
        )

        ###
        # States and state history
        ###

        # get current state and use it to transform the previous and following states into the local robot frame
        # shape: [N, 7] with [x, y, z, qx, qy, qz, qw]
        initial_states = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], 0][:, None, :7]
        initial_states_SE3 = pp.SE3(initial_states.repeat(1, self.model_cfg.prediction_horizon, 1).reshape(-1, 7))

        # get the state history
        self.state_history = self.state_history_transformer(
            replay_buffer,
            start_idx,
            initial_states,
            self.model_cfg.history_length,
            self.model_cfg.exclude_state_idx_from_input,
        )

        # get the future positions along the trajectory
        states = torch.concatenate(
            [
                replay_buffer.states[start_idx[:, 0], start_idx[:, 1] + idx + 1, 0][:, None]
                for idx in range(self.model_cfg.prediction_horizon)
            ],
            dim=1,
        )
        states_SE3 = pp.SE3(states[..., :7].reshape(-1, 7))
        states_SE3 = (pp.Inv(initial_states_SE3) * states_SE3).tensor()
        states_yaw = math_utils.euler_xyz_from_quat(states_SE3[..., [6, 3, 4, 5]])[2]
        # rotation encoded as [sin(yaw), cos(yaw)] to avoid jump in representation
        # Check: Learning with 3D rotations, a hitchhiker’s guide to SO(3), 2024, Frey et al.
        states_yaw_sin_cos = torch.stack([torch.sin(states_yaw), torch.cos(states_yaw)], dim=1)
        # final state: [N, Prediction Horizon, 3 (pos) + 2 (yaw) + 1 (collision) + rest of the state]
        self.states = torch.concatenate(
            [
                states_SE3.reshape(-1, self.model_cfg.prediction_horizon, 7)[..., :2],
                states_yaw_sin_cos.reshape(-1, self.model_cfg.prediction_horizon, 2),
                states[..., 7:],
            ],
            dim=2,
        )
        if self.model_cfg.hard_contact_metric == "contact" or self.model_cfg.hard_contact_metric == "torque":
            max_metric = torch.concatenate(
                [
                    replay_buffer.states[start_idx[:, 0], start_idx[:, 1] + idx + 1, :, 8].unsqueeze(1)
                    for idx in range(self.model_cfg.prediction_horizon)
                ],
                dim=1,
            )
            max_metric = torch.max(max_metric, dim=-1)[0]
            if self.model_cfg.hard_contact_metric == "contact":
                self.states[..., 5] = torch.log(max_metric)
            else:
                self.states[..., 5] = max_metric

        ###
        # Observations
        ###

        self.obs_proprioceptive = replay_buffer.observations_proprioceptive[start_idx[:, 0], start_idx[:, 1]]
        if replay_buffer.observations_exteroceptive is not None:
            self.obs_exteroceptive = replay_buffer.observations_exteroceptive[start_idx[:, 0], start_idx[:, 1]]
        else:
            self.obs_exteroceptive = None
        if replay_buffer.add_observations_exteroceptive is not None:
            self.add_obs_exteroceptive = replay_buffer.add_observations_exteroceptive[start_idx[:, 0], start_idx[:, 1]]
        else:
            self.add_obs_exteroceptive = None

        ###
        # Perfect velocity following - EVALUATION ONLY
        ###

        # get the resulting change in position and angle when applying the commands perfectly
        # velocity command units x: [m/s], y: [m/s], phi: [rad/s]
        perfect_velocity_following_individual_frame = self.actions * self.model_cfg.command_timestep

        # Cumsum is an inplace operation therefore the clone is necesasry
        cummulative_yaw = perfect_velocity_following_individual_frame.clone()[..., -1].cumsum(-1)

        # We need to take the non-linearity by the rotation into account
        r_vec1 = torch.stack([torch.cos(cummulative_yaw), -torch.sin(cummulative_yaw)], dim=-1)
        r_vec2 = torch.stack([torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=-1)
        so2 = torch.stack([r_vec1, r_vec2], dim=2)

        # Move the rotation in time and fill first timestep with identity - see math chapter
        so2 = torch.roll(so2, shifts=1, dims=1)
        so2[:, 0, :, :] = torch.eye(2, device=so2.device)[None].repeat(so2.shape[0], 1, 1)

        actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ perfect_velocity_following_individual_frame[
            ..., :2
        ].contiguous().reshape(-1, 2, 1)
        actions_local_frame = actions_local_frame.contiguous().reshape(so2.shape[0], so2.shape[1], 2)
        cumulative_position = (actions_local_frame).cumsum(-2)
        self.perfect_velocity_following_local_frame = torch.cat(
            [cumulative_position, torch.sin(cummulative_yaw)[:, :, None], torch.cos(cummulative_yaw)[:, :, None]],
            dim=-1,
        )

        ###
        # Filter data
        ###
        # init keep index array
        keep_idx = torch.ones(
            self.state_history.shape[0], dtype=torch.bool, device=self.replay_buffer_cfg.buffer_device
        )

        # filter every sample with collision in the first three steps (get collision state from samples)
        collision_env, collision_idx = torch.where(self.states[..., 4])
        remove_idx = collision_env[collision_idx < self.cfg.sample_filter_first_steps_coll]
        remove_idx = torch.unique(remove_idx)
        keep_idx[remove_idx] = False
        # filter every sample that has a collision in its initial position
        collision_env = torch.where(self.state_history[:, 0, 4])[0]
        keep_idx[collision_env] = False
        # restrict ratio of samples that move less than 1m in the entire trajectory to 10%
        if self.cfg.small_motion_ratio is not None:
            small_movement_idx = torch.where(
                torch.norm(torch.abs(self.states[:, -1, :2]), dim=1) < self.cfg.small_motion_threshold
            )[0]
            small_movement_ratio = small_movement_idx.shape[0] / self.states.shape[0]
            if small_movement_ratio > self.cfg.small_motion_ratio:
                small_movement_idx = small_movement_idx[: int(self.states.shape[0] * self.cfg.small_motion_ratio)]
                keep_idx[small_movement_idx] = False

        # filter samples with too little height difference
        if self.cfg.height_threshold is not None:
            state_height = states_SE3[..., 2].reshape(-1, self.model_cfg.prediction_horizon)
            height_diff = torch.max(torch.abs(state_height[:, 1:] - state_height[:, :-1]), dim=-1)[0]
            keep_idx[torch.where(height_diff < self.cfg.height_threshold)[0]] = False
            print(
                "[INFO] Filtered samples with too little height difference! Overall"
                f" {(height_diff < self.cfg.height_threshold).sum().item()} samples filtered!"
            )

        # filter samples
        initial_states = initial_states.repeat(1, self.model_cfg.prediction_horizon, 1)[keep_idx]
        states = states[keep_idx]
        self._filter_idx(keep_idx)

        ###
        # Collision handling - repeat last state when in collision
        ###

        # for states that are in collision, take the last state and copy it to the rest of the trajectory
        collision_env, collision_idx = torch.where(self.states[..., 4])
        if len(collision_env) > 0:
            # if there are multiple collisions within the sampled trajectory, only use the first one
            collision_env_red = torch.unique(collision_env)
            collision_idx_red = torch.hstack(
                [torch.min(collision_idx[collision_env == curr_env]) for curr_env in collision_env_red]
            )
            # get indices
            indices = [
                [
                    collision_env_red[idx].repeat(self.model_cfg.prediction_horizon - collision_idx_red[idx]),
                    torch.arange(
                        collision_idx_red[idx],
                        self.model_cfg.prediction_horizon,
                        device=self.replay_buffer_cfg.buffer_device,
                    ),
                    collision_idx_red[idx].repeat(self.model_cfg.prediction_horizon - collision_idx_red[idx]),
                ]
                for idx in range(len(collision_env_red))
            ]
            env_idx = torch.hstack([curr_indices[0] for curr_indices in indices])
            horizon_idx = torch.hstack([curr_indices[1] for curr_indices in indices])
            command_idx = torch.hstack([curr_indices[2] for curr_indices in indices])
            # update data
            self.states[env_idx, horizon_idx] = self.states[env_idx, command_idx]

            # NOTE: actions should not be copied, otherwise model learns to recognize collision from actions
            # self.actions[env_idx, horizon_idx] = self.actions[env_idx, command_idx]

            # NOTE: for the evaluation, also the perfect velocity is not corrected
            # self.perfect_velocity_following_local_frame[env_idx, horizon_idx] = (
            #     self.perfect_velocity_following_local_frame[env_idx, command_idx]
            # )

        # statistics of generated data
        max_distance = torch.norm(self.states[:, -1, :2], dim=1)

        # filter outliers, i.e. due to falling from the ground plane
        outlier_idx = torch.where(
            torch.logical_or(
                torch.any(torch.any(torch.abs(self.states[..., :3]) > 10, dim=-1), dim=-1), max_distance > 10.0
            )
        )[0]

        if len(outlier_idx) > 0:
            print("[WARNING] Found outliers with max position > 10.0!")
            keep_idx = torch.ones(
                self.state_history.shape[0], dtype=torch.bool, device=self.replay_buffer_cfg.buffer_device
            )
            keep_idx[outlier_idx] = False
            initial_states = initial_states[keep_idx]
            max_distance = max_distance[keep_idx]
            states = states[keep_idx]
            self._filter_idx(keep_idx)

        ###
        # Normalize the hard contact observation
        ###

        # get min and max torque
        self.max_hard_contact_obs = torch.max(torch.max(self.states[..., 5]), self.max_hard_contact_obs)
        self.min_hard_contact_obs = torch.min(torch.min(self.states[..., 5]), self.min_hard_contact_obs)

        # normalize torque
        self.state_history[..., 5] = (self.state_history[..., 5] - self.min_hard_contact_obs) / (
            self.max_hard_contact_obs - self.min_hard_contact_obs
        )
        self.states[..., 5] = (self.states[..., 5] - self.min_hard_contact_obs) / (
            self.max_hard_contact_obs - self.min_hard_contact_obs
        )

        ###
        # Extract maximum physical values of the system to constrain model
        ###

        # get the maximum observed velocity
        lin_velocity = torch.abs((self.states[:, 1:, :2] - self.states[:, :-1, :2]) / self.model_cfg.command_timestep)
        heading = torch.atan2(self.states[:, :, 2], self.states[:, :, 3])
        # enforce periodicity of the heading
        yaw_diff = torch.abs(heading[:, 1:] - heading[:, :-1])
        yaw_diff = math_utils.wrap_to_pi(yaw_diff)
        ang_velocity = torch.abs(yaw_diff / self.model_cfg.command_timestep)
        max_velocity = torch.concatenate(
            [torch.max(lin_velocity.reshape(-1, 2), dim=0)[0], torch.max(ang_velocity.reshape(-1, 1), dim=0)[0]], dim=0
        )

        # get the maximum observed acceleration
        max_lin_acceleration = torch.max(
            torch.abs((lin_velocity[:, 1:] - lin_velocity[:, :-1]) / self.model_cfg.command_timestep).reshape(-1, 2),
            dim=0,
        )[0]
        max_ang_acceleration = torch.max(
            torch.abs((ang_velocity[:, 1:] - ang_velocity[:, :-1]) / self.model_cfg.command_timestep).reshape(-1, 1),
            dim=0,
        )[0]
        max_acceleration = torch.concatenate([max_lin_acceleration, max_ang_acceleration], dim=0)

        # check the maximum velocity is not more than the maximum commanded velocity
        max_possible_velocity = (VEL_RANGE_X[1] ** 2 + VEL_RANGE_Y[1] ** 2) ** 0.5
        collision_samples = self.states[..., 4].any(dim=1)
        max_velocity_non_collision = torch.concatenate(
            [
                torch.max(lin_velocity[~collision_samples].reshape(-1, 2), dim=0)[0],
                torch.max(ang_velocity[~collision_samples].reshape(-1, 1), dim=0)[0],
            ],
            dim=0,
        )
        if torch.any(
            max_velocity_non_collision
            > torch.tensor([max_possible_velocity, max_possible_velocity, VEL_RANGE_YAW[1]]) * 1.1
        ):
            # NOTE: When the robot is "falling" the velocity, especially the angular velocity, can be very high
            #       while the robot does not necessary collide
            non_colliding_vels = torch.concatenate(
                [lin_velocity[~collision_samples], ang_velocity[~collision_samples].unsqueeze(-1)], dim=-1
            )
            exceeding_cases = torch.any(
                non_colliding_vels
                > torch.tensor([[[max_possible_velocity, max_possible_velocity, VEL_RANGE_YAW[1]]]]) * 1.1,
                dim=-1,
            )
            # get the z diff for the exceeding cases
            z_diff = states[~collision_samples, 1:, 2] - states[~collision_samples, :-1, 2]
            z_diff_exceeding = z_diff[exceeding_cases]
            print(
                f"[WARNING] Maximum observed velocity {max_velocity_non_collision.cpu().tolist()} is higher in"
                f" {exceeding_cases.sum().item()} cases than the maximum commanded velocity"
                f" {[max_possible_velocity, max_possible_velocity, VEL_RANGE_YAW[1]]}! In"
                f" {(torch.abs(z_diff_exceeding) > 0.1).sum().item()} cases the z diff is larger than 0.1m!"
            )

            # restrict maximum applied velocity to the maximum observed velocity for non-collision cases and cases
            # without a jump in z coordinate
            if torch.any(~exceeding_cases.any(dim=1)):
                max_velocity_applied = non_colliding_vels[~exceeding_cases.any(dim=1)].reshape(-1, 3).max(dim=0)[0]
                non_colliding_vels = torch.abs(non_colliding_vels)
                max_acceleration_applied = torch.max(
                    torch.abs(
                        (
                            non_colliding_vels[~exceeding_cases.any(dim=1)][:, 1:]
                            - non_colliding_vels[~exceeding_cases.any(dim=1)][:, :-1]
                        )
                        / self.model_cfg.command_timestep
                    ).reshape(-1, 3),
                    dim=0,
                )[0]
            else:
                max_velocity_applied = max_velocity.clone()
                max_acceleration_applied = max_acceleration.clone()
        else:
            max_velocity_applied = max_velocity.clone()
            max_acceleration_applied = max_acceleration.clone()

        # scale the applied limits with a safety factor of 5% to allow for larger corrections
        max_velocity_applied *= 1.05
        max_acceleration_applied *= 1.05

        ###
        # Extract further statistics
        ###

        # compare states and perfect veloicty estimate
        pos_diff = torch.norm(self.states[..., :2] - self.perfect_velocity_following_local_frame[..., :2], dim=-1)
        cummulative_yaw_states = torch.atan2(self.states[..., 2], self.states[..., 3])
        cummulative_yaw_perfect_velocity_following = torch.atan2(
            self.perfect_velocity_following_local_frame[..., 2], self.perfect_velocity_following_local_frame[..., 3]
        )
        yaw_diff = torch.abs(cummulative_yaw_states - cummulative_yaw_perfect_velocity_following)
        # account for the periodicity of the yaw
        yaw_diff = math_utils.wrap_to_pi(yaw_diff)

        # Mean and Varatity of actions
        action_var = self.actions.view(-1, 3).std(dim=0)
        action_mean = self.actions.view(-1, 3).mean(dim=0)

        ###
        # Check for nan and inf values
        ###

        if torch.any(torch.isnan(self.states)) or torch.any(torch.isinf(self.states)):
            raise ValueError("Nan/ Inf values in states!")
        if torch.any(torch.isnan(self.state_history)) or torch.any(torch.isinf(self.state_history)):
            raise ValueError("Nan/ Inf values in state history!")
        if torch.any(torch.isnan(self.obs_proprioceptive)) or torch.any(torch.isinf(self.obs_proprioceptive)):
            raise ValueError("Nan/ Inf values in proprioceptive observations!")
        if self.obs_exteroceptive is not None and (
            torch.any(torch.isnan(self.obs_exteroceptive)) or torch.any(torch.isinf(self.obs_exteroceptive))
        ):
            raise ValueError("Nan/ Inf values in exteroceptive observations!")
        if self.add_obs_exteroceptive is not None and (
            torch.any(torch.isnan(self.add_obs_exteroceptive)) or torch.any(torch.isinf(self.add_obs_exteroceptive))
        ):
            raise ValueError("Nan/ Inf values in additional exteroceptive observations!")
        if torch.any(torch.isnan(self.actions)) or torch.any(torch.isinf(self.actions)):
            raise ValueError("Nan/ Inf values in actions!")
        if torch.any(torch.isnan(self.perfect_velocity_following_local_frame)) or torch.any(
            torch.isinf(self.perfect_velocity_following_local_frame)
        ):
            raise ValueError("Nan/ Inf values in perfect velocity following!")

        ###
        # Ablation studies
        ###

        if self.cfg.ablation_no_state_obs:
            self.state_history *= 0.0
        elif self.cfg.ablation_no_proprio_obs:
            self.obs_proprioceptive *= 0.0
        elif self.cfg.ablation_no_height_scan and self.obs_exteroceptive is not None:
            self.obs_exteroceptive *= 0.0

        ###
        # Print meta information
        ###

        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"
        table.align["Value"] = "r"

        # Add rows with formatted values
        table.add_row(("Average max distance", f"{torch.mean(torch.abs(max_distance), dim=0).item():.4f}"))
        table.add_row(("Average collision rate", f"{self.collision_rate:.4f}"))
        table.add_row(("Max velocity", [f"{v:.4f}" for v in max_velocity.cpu().tolist()]))
        table.add_row(("Max acceleration", [f"{a:.4f}" for a in max_acceleration.cpu().tolist()]))
        table.add_row(("Max velocity applied", [f"{v:.4f}" for v in max_velocity_applied.cpu().tolist()]))
        table.add_row(("Max acceleration applied", [f"{a:.4f}" for a in max_acceleration_applied.cpu().tolist()]))
        table.add_row(("Max hard contact observation", f"{self.max_hard_contact_obs.item():.4f}"))
        table.add_row(("Min hard contact observation", f"{self.min_hard_contact_obs.item():.4f}"))

        # Print distance percentages
        for distance in range(1, int(torch.max(torch.abs(max_distance)).item()) + 2):
            ratio = (
                torch.sum(torch.all(torch.vstack((max_distance > distance - 1, max_distance < distance)), dim=0))
                / self.states.shape[0]
            )
            table.add_row((f"Ratio between {distance - 1} - {distance}m", f"{ratio.item():.4f}"))
        for distance in range(1, int(torch.ceil(torch.max(torch.abs(self.states[:, -1, 0])))) + 1):
            ratio = (
                torch.sum(
                    torch.all(
                        torch.vstack((
                            torch.abs(self.states[:, -1, 0]) > distance - 1,
                            torch.abs(self.states[:, -1, 0]) < distance,
                        )),
                        dim=0,
                    )
                )
                / self.states.shape[0]
            )
            table.add_row((f"Ratio between {distance - 1} - {distance}m in x", f"{ratio.item():.4f}"))
        for distance in range(1, int(torch.ceil(torch.max(torch.abs(self.states[:, -1, 1])))) + 1):
            ratio = (
                torch.sum(
                    torch.all(
                        torch.vstack((
                            torch.abs(self.states[:, -1, 1]) > distance - 1,
                            torch.abs(self.states[:, -1, 1]) < distance,
                        )),
                        dim=0,
                    )
                )
                / self.states.shape[0]
            )
            table.add_row((f"Ratio between {distance - 1} - {distance}m in y", f"{ratio.item():.4f}"))

        # Print differences between states and perfect velocity following
        table.add_row((
            "Perf Vel Position difference",
            f"{torch.mean(pos_diff).item():.4f}" + " \u00b1 " + f"{torch.std(pos_diff).item():.4f}",
        ))
        table.add_row((
            "Perf Vel Yaw difference",
            f"{torch.mean(yaw_diff).item():.4f}" + " \u00b1 " + f"{torch.std(yaw_diff).item():.4f}",
        ))
        table.add_row(("Perf Vel Max position difference", f"{torch.max(pos_diff).item():.4f}"))
        table.add_row(("Perf Vel Max yaw difference", f"{torch.max(yaw_diff).item():.4f}"))

        # Print action variance
        table.add_row(("Action Mean", [f"{v:.4f}" for v in action_mean.cpu().tolist()]))
        table.add_row(("Action Variance", [f"{v:.4f}" for v in action_var.cpu().tolist()]))

        # add info about ablation studies
        table.add_row(("Ablation no state obs", self.cfg.ablation_no_state_obs))
        table.add_row(("Ablation no proprio obs", self.cfg.ablation_no_proprio_obs))
        table.add_row(("Ablation no height scan", self.cfg.ablation_no_height_scan))

        # Print table
        print(f"[INFO] Dataset Metrics {self.states.shape[0]} samples\n", table)

        if False:
            ###
            # Debug try to crop height scan to current position
            ###

            # visualize the split up height scane for each step of the FDM
            print("Debugging")
            import math

            height_scan_res = 0.1
            # Define the bounds of the subregion to extract
            x_min, x_max = -0.5, 1.0  # height --> x
            y_min, y_max = -1.0, 1.0  # width --> y

            height_scan_shape = (self.obs_exteroceptive.shape[-2], self.obs_exteroceptive.shape[-1])
            height_scan_robot_center = [height_scan_shape[0] / 2, 0.5 / height_scan_res]

            # get effective translation
            # since in robot frame, the y translation is against the height axis x direction, has to be negative
            effective_translation_tensor_x = (
                -self.states[:, :, 1].reshape(-1) / height_scan_res + height_scan_robot_center[0]
            )
            effective_translation_tensor_y = (
                self.states[:, :, 0].reshape(-1) / height_scan_res + height_scan_robot_center[1]
            )

            # Create a meshgrid of coordinates
            idx_tensor_x, idx_tensor_y = torch.meshgrid(
                torch.arange(y_min / height_scan_res, (y_max / height_scan_res) + 1),
                torch.arange(x_min / height_scan_res, (x_max / height_scan_res) + 1),
                indexing="ij",
            )
            idx_tensor_x = idx_tensor_x.flatten().float().repeat(self.states.shape[0] * self.states.shape[1], 1)
            idx_tensor_y = idx_tensor_y.flatten().float().repeat(self.states.shape[0] * self.states.shape[1], 1)

            # angle definition for the height scan coordinate system is opposite of the tensor system, so negative
            s = self.states[:, :, 2].reshape(-1).unsqueeze(1)
            c = self.states[:, :, 3].reshape(-1).unsqueeze(1)
            idx_crop_x = (c * idx_tensor_x - s * idx_tensor_y + effective_translation_tensor_x.unsqueeze(1)).int()
            idx_crop_y = (s * idx_tensor_x + c * idx_tensor_y + effective_translation_tensor_y.unsqueeze(1)).int()

            # move idx tensors of the new image to 0,0 in upper left corner
            idx_tensor_x += torch.abs(torch.min(idx_tensor_x, dim=-1)[0]).unsqueeze(1)
            idx_tensor_y += torch.abs(torch.min(idx_tensor_y, dim=-1)[0]).unsqueeze(1)

            # filter_idx outside the image
            filter_idx = (
                (idx_crop_x >= 0)
                & (idx_crop_x < height_scan_shape[0])
                & (idx_crop_y >= 0)
                & (idx_crop_y < height_scan_shape[1])
            )
            idx_crop_x[~filter_idx] = 0
            idx_crop_y[~filter_idx] = 0

            new_image = torch.zeros((
                self.states.shape[0] * self.states.shape[1],
                math.ceil((y_max - y_min) / height_scan_res + 1),
                math.ceil((x_max - x_min) / height_scan_res + 1),
            ))
            ALL_INDICES = torch.arange(self.states.shape[0] * self.states.shape[1]).int()[:, None].repeat(1, 336)
            new_image[ALL_INDICES, idx_tensor_x.int(), idx_tensor_y.int()] = self.obs_exteroceptive.repeat(
                1, self.states.shape[1], 1, 1
            ).reshape(-1, *height_scan_shape)[ALL_INDICES, idx_crop_x.int(), idx_crop_y.int()]

            filter_idx_nonzero = (~filter_idx).nonzero()
            new_image[
                filter_idx_nonzero[:, 0].int(),
                idx_tensor_x[filter_idx_nonzero[:, 0], filter_idx_nonzero[:, 1]].int(),
                idx_tensor_y[filter_idx_nonzero[:, 0], filter_idx_nonzero[:, 1]].int(),
            ] = -1

            import matplotlib.pyplot as plt

            # Visualization using matplotlib
            idx = 1
            fig, axs = plt.subplots(2, 11, figsize=(55, 10))

            vmin = -1
            vmax = torch.max(self.obs_exteroceptive[idx, 0]).item()

            img = axs[0, 0].imshow(self.obs_exteroceptive[idx, 0].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            axs[0, 0].set_title("Large Height Scan")
            axs[0, 0].set_xlabel("X")
            axs[0, 0].set_ylabel("Y")

            for i in range(10):
                print(i)
                axs[0, i + 1].imshow(
                    new_image[idx * self.states.shape[1] + i].numpy(), cmap="viridis", vmin=vmin, vmax=vmax
                )
                axs[0, i + 1].set_title(
                    f"{i}:"
                    f" {self.states[idx, i, 0].float():.4f} {self.states[idx, i, 1].float():.4f} {torch.atan2(self.states[idx, i, 2], self.states[idx, i, 3]).float():.4f}"
                )
                axs[0, i + 1].set_xlabel("X")
                axs[0, i + 1].set_ylabel("Y")

                mask = torch.zeros(*height_scan_shape, dtype=torch.bool)
                mask[idx_crop_x[idx * self.states.shape[1] + i], idx_crop_y[idx * self.states.shape[1] + i]] = True
                masked_image = torch.where(mask, self.obs_exteroceptive[idx, 0], torch.tensor(-1))

                axs[1, i + 1].imshow(masked_image.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
                axs[1, i + 1].set_xlabel("X")
                axs[1, i + 1].set_ylabel("Y")

            # Create a colorbar
            cbar = fig.colorbar(img, ax=axs, fraction=0.02, pad=0.04)
            cbar.set_label("Color Scale")

            plt.tight_layout()
            plt.savefig("height_scan.png")

        return initial_states, max_velocity_applied, max_acceleration_applied

    """
    Private functions
    """

    def _sample_random_traj_idx(self, replay_buffer: ReplayBuffer):
        # sample random start indexes
        # collision not correctly captured at the last entry of the trajectory, exclude it from trajectories
        start_idx = torch.randint(
            1,
            self.replay_buffer_cfg.trajectory_length - self.model_cfg.prediction_horizon - 1,
            (self.cfg.num_samples,),
            device=self.replay_buffer_cfg.buffer_device,
        )
        env_idx = torch.randint(
            0, replay_buffer.env.num_envs, (self.cfg.num_samples,), device=self.replay_buffer_cfg.buffer_device
        )
        return torch.vstack([env_idx, start_idx]).T

    def _sample_collision_traj(self, replay_buffer: ReplayBuffer):
        # identify collision samples (at least self.model_cfg.prediction_horizon-2 steps before the end to sample commands)
        collision_samples = torch.where(replay_buffer.states[:, 1 : -self.model_cfg.prediction_horizon + 2, :, 7])
        # sample start position before the collision, should be at least two steps before the collision
        collision_start_idx = torch.randint(
            2,
            self.model_cfg.prediction_horizon,
            (collision_samples[0].shape[0],),
            device=self.replay_buffer_cfg.buffer_device,
        )
        # clip start position
        collision_start_idx = torch.clip(collision_samples[1] - collision_start_idx, 0)
        # get trajectory start indexes
        return torch.vstack([collision_samples[0], collision_start_idx]).T

    def _filter_idx(self, keep_idx: torch.Tensor):
        """Filter data and only keep the given indexes. After filtering, update the number of samples"""
        # filter data
        self.state_history = self.state_history[keep_idx]
        self.obs_proprioceptive = self.obs_proprioceptive[keep_idx]
        self.actions = self.actions[keep_idx]
        self.states = self.states[keep_idx]
        self.perfect_velocity_following_local_frame = self.perfect_velocity_following_local_frame[keep_idx]
        if self.obs_exteroceptive is not None:
            self.obs_exteroceptive = self.obs_exteroceptive[keep_idx]
        if self.add_obs_exteroceptive is not None:
            self.add_obs_exteroceptive = self.add_obs_exteroceptive[keep_idx]

        # update sample number
        self._actual_nbr_samples = torch.sum(keep_idx).item()

    """
    Static helper functions
    """

    @staticmethod
    def state_history_transformer(
        replay_buffer: ReplayBuffer,
        start_idx: torch.Tensor,
        initial_states: torch.Tensor,
        history_length: int,
        exclude_index: list[int] | None = None,
    ):
        """transform the state history into the local robot frame

        Individual function as also used for evaluation call when the model should only do predictions.
        """
        # repeat initial state to match the state history
        initial_states_SE3 = pp.SE3(initial_states.repeat(1, history_length, 1).reshape(-1, 7))
        # transform the state history into the local robot frame
        state_history = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], :, :7]
        state_history = pp.SE3(state_history.reshape(-1, 7))
        state_history_local = (pp.Inv(initial_states_SE3) * state_history).tensor()
        state_history_pos = state_history_local.reshape(-1, history_length, 7)[..., :2]
        state_history_yaw = math_utils.euler_xyz_from_quat(state_history_local[..., [6, 3, 4, 5]])[2]
        # rotation encoded as [sin(yaw), cos(yaw)] to avoid jump in representation
        # Check: Learning with 3D rotations, a hitchhiker’s guide to SO(3), 2024, Frey et al.
        state_history_yaw = torch.stack([torch.sin(state_history_yaw), torch.cos(state_history_yaw)], dim=1)
        state_history_yaw = state_history_yaw.reshape(-1, history_length, 2)
        # get the rest of the state and potentially exclude some indices
        rest_of_state = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], :, 7:]
        if exclude_index is not None:
            keep_idx = torch.ones(replay_buffer.states.shape[-1], device=replay_buffer.states.device, dtype=torch.bool)
            keep_idx[exclude_index] = False
            keep_idx = keep_idx[7:]
            rest_of_state = rest_of_state[..., keep_idx]
        # final state history: [N, History Length, 3 (pos) + 2 (yaw) + 1 (collision) + rest of the state]
        return torch.concatenate([state_history_pos, state_history_yaw, rest_of_state], dim=2)

    """
    Properties called when accessing the data
    """

    def __len__(self):
        return self._actual_nbr_samples

    def __getitem__(self, index: int):
        # get extereoceptive and apply noise model
        if self.obs_exteroceptive is not None and self.extereoceptive_noise_model is None:
            exteroceptive = self.obs_exteroceptive[index].type(torch.float32)
        elif self.obs_exteroceptive is not None:
            exteroceptive = self.extereoceptive_noise_model(self.obs_exteroceptive[index].type(torch.float32))
        else:
            exteroceptive = torch.zeros(1)

        # get additional exteroceptive observation
        if self.add_obs_exteroceptive is not None:
            add_exteroceptive = self.add_obs_exteroceptive[index].type(torch.float32)
        else:
            add_exteroceptive = torch.zeros(1)

        return (
            # model inputs
            self.state_history[index],
            self.obs_proprioceptive[index],
            exteroceptive,
            self.actions[index],
            add_exteroceptive,
            # model targets
            self.states[index],
            # eval data
            self.perfect_velocity_following_local_frame[index],
        )
