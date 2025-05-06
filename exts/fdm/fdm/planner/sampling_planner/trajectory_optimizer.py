# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import os
import pickle
import subprocess
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from typing import TYPE_CHECKING

from skimage.draw import line

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.timer import Timer

from nav_suite.terrain_analysis import TerrainAnalysis

from fdm import FDM_DATA_DIR
from fdm.env_cfg import TERRAIN_ANALYSIS_CFG

from .robot_shape import get_robot_shape
from .trajectory_optimizer_cfg import ActionCfg, RobotCfg, TrajectoryOptimizerCfg
from .trajectory_optimizer_mbrl import BatchedICEMOptimizer, BatchedMPPIOptimizer
from .utils import (
    TraversabilityFilter,
    cosine_distance,
    get_non_zero_action_length,
    get_se2,
    get_x_y_yaw,
    smallest_angle,
    state_history_transformer,
)

if TYPE_CHECKING:
    from fdm.model import FDMModel


class SimpleSE2TrajectoryOptimizer:
    def __init__(
        self,
        action_cfg: ActionCfg,
        robot_cfg: RobotCfg,
        optim: BatchedICEMOptimizer | BatchedMPPIOptimizer,
        to_cfg: TrajectoryOptimizerCfg,
        device: torch.device,
    ):
        """
        Initializes the SimpleSE2TrajectoryOptimizer with the given configurations and device.

        Args:
            action_cfg (ActionCfg): Configuration for the action space.
            robot_cfg (RobotCfg): Configuration for the robot footprint.
            to_cfg (TrajectoryOptimizerCfg): Configuration for the trajectory optimizer.
            optim (TODO): Underlying Black box optimizer
            device (torch.device): The device (CPU/GPU) on which to perform the computations.
        """

        self.action_cfg = action_cfg
        self.to_cfg = to_cfg
        self.device = device
        self.frame_id = "odom"
        self.fatal_xy, self.risky_xy, self.cautious_xy = get_robot_shape(robot_cfg, device)

        # fdm parameters
        self.fdm_model: FDMModel | None = None
        self.terrain_analysis: TerrainAnalysis | None = None

        # Initialize Optimizer
        self.optim = optim
        self.previous_solution = torch.zeros(
            self.optim.batch_size, self.optim.planning_horizon, self.optim.action_dimension, device=self.device
        )

        if self.to_cfg.states_cost_w_cost_map:
            weight_file = subprocess.getoutput(
                'echo "' + os.path.join(FDM_DATA_DIR, "Traversability-Model", "weights.dat") + '"'
            )
            with open(weight_file, "rb") as file:
                weights = pickle.load(file)
            self.traversability_filter = TraversabilityFilter(
                weights["conv1.weight"], weights["conv2.weight"], weights["conv3.weight"], weights["conv_final.weight"]
            )
            self.traversability_filter.to(self.device).eval()

        # Set objective function
        if self.to_cfg.n_step_fwd:
            self.func = self.b_obj_func_N_step
        else:
            self.func = self.b_obj_func

        self.debug_info = {}

    def set_fdm_classes(self, fdm_model: FDMModel, env: ManagerBasedRLEnv):
        self.fdm_model = fdm_model
        self.terrain_analysis = TerrainAnalysis(cfg=TERRAIN_ANALYSIS_CFG, scene=env.scene)
        self.terrain_analysis.analyse()
        self.height_scan_resolution = getattr(env.scene.sensors["env_sensor"].cfg.pattern_cfg, "resolution", 1.0)
        self.height_scan_size = getattr(env.scene.sensors["env_sensor"].cfg.pattern_cfg, "size", (10.0, 10.0))
        self.height_scan_offset = env.scene.sensors["env_sensor"].cfg.offset.pos

    ###
    # Operations
    ###

    def plan(
        self, obs: dict, env_ids: torch.Tensor | None = None, return_states: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Initializes the observation dictionary with default values for planning.

        Args:
            obs (dict): A dictionary containing the following key-value pairs:
                - "goal": (torch.tensor, shape:=(BS, 3)): representing the goal with (x,y,yaw) in the odom frame.
                - "resample_population": bool: If the population should be resampled
                - "start": (torch.tensor, shape:=(BS, 3)): representing the start with (x,y,yaw) in the odom frame.
            env_ids: The environment ids for which to plan.
        Returns:
            torch.Tensor: The planned trajectory with shape (BS, TRAJ_LENGTH, STATE_DIM).
            torch.Tensor: The planned velocity with shape (BS, TRAJ_LENGTH, CONTROL_DIM).

        """
        # save obs for later use
        self.obs = obs

        if env_ids is None:
            BS = self.obs["start"].shape[0]
            self.env_ids = slice(None)
        else:
            BS = len(env_ids)
            self.env_ids = env_ids
            assert BS != 0, "No environments to plan for. This case should be handled by the planner."

        # resample goal position if its out the height scan range (out of distribution for the training set)
        # -- check if outside the height scan range
        # start_state = self.get_start_state(1, 1)[:, 0].clone()
        # se2_odom_base = get_se2(start_state)  # Transformation from base to odom frame
        # se2_odom_goal = get_se2(self.obs["goal"])  # Transformation from goal to odom frame
        # se2_base_goal = torch.inverse(se2_odom_base) @ se2_odom_goal  # Transformation from base to goal frame
        # xyyaw_base_goal = get_x_y_yaw(se2_base_goal)
        # goal_outside_range = torch.logical_or(
        #     torch.abs(xyyaw_base_goal[..., 0]) > (self.height_scan_size[0]/ 2 + self.height_scan_offset[0]),
        #     torch.abs(xyyaw_base_goal[..., 1]) > (self.height_scan_size[1] / 2 + self.height_scan_offset[1]),
        # )
        # if torch.any(goal_outside_range):
        #     # -- each side of the height scan is descrited into n points, compute the minimal distance between goal and
        #     #    the height scan points, then select the n clostest points as goal points
        #     n_sample_points = 5
        #     x_sample_points = torch.linspace(-self.height_scan_size[0]/ 2 + self.height_scan_offset[0], self.height_scan_size[0]/ 2 + self.height_scan_offset[0], n_sample_points, device=self.device)
        #     y_sample_points = torch.linspace(-self.height_scan_size[1]/ 2 + self.height_scan_offset[1], self.height_scan_size[1]/ 2 + self.height_scan_offset[1], n_sample_points, device=self.device)
        #     # NOTE: corner points are duplicated, so we remove them
        #     height_scan_points = torch.concatenate((
        #         torch.stack((x_sample_points, torch.ones(n_sample_points, device=self.device) * -self.height_scan_size[1]/ 2 + self.height_scan_offset[1]), dim=1),
        #         torch.stack((x_sample_points, torch.ones(n_sample_points, device=self.device) * self.height_scan_size[1]/ 2 + self.height_scan_offset[1]), dim=1),
        #         torch.stack((torch.ones(n_sample_points, device=self.device) * -self.height_scan_size[0]/ 2 + self.height_scan_offset[0], y_sample_points), dim=1)[1 : -1],
        #         torch.stack((torch.ones(n_sample_points, device=self.device) * self.height_scan_size[0]/ 2 + self.height_scan_offset[0], y_sample_points), dim=1)[1 : -1],
        #     ), dim=0)
        #     distance = xyyaw_base_goal[goal_outside_range][:, None, :2] - height_scan_points[None, :, :]
        #     distance = torch.norm(distance, dim=-1)
        #     closest_points_idx = torch.argsort(distance, dim=1)[:, :n_sample_points]
        #     closest_points = height_scan_points[closest_points_idx.flatten()]
        #     # -- transform the clostest points in the odom frame
        #     se2_clostest_points_base = get_se2(torch.concatenate((closest_points, torch.zeros_like(closest_points[..., 0])[..., None]), dim=-1))
        #     se2_clostest_points_odom = se2_odom_base[goal_outside_range, None, :, :].repeat(1, n_sample_points, 1, 1).reshape(-1, 3, 3) @ se2_clostest_points_base
        #     closest_points_odom = get_x_y_yaw(se2_clostest_points_odom)
        #     closest_points_odom[..., 2] *= 0.0
        #     # -- exchange the goal position, if the goal is outside the height scan range and repeat the env_ids and start_points
        #     self.obs["start"] = torch.cat((self.obs["start"][~goal_outside_range], self.obs["start"][goal_outside_range][:, None, :].repeat(1, n_sample_points, 1).view(-1, 3)), dim=0)
        #     self.obs["goal"] = torch.cat((self.obs["goal"][~goal_outside_range], closest_points_odom), dim=0)
        #     env_ids = torch.tensor(self.env_ids, dtype=torch.long, device=self.device)
        #     env_ids = torch.cat((env_ids[~goal_outside_range], env_ids[goal_outside_range][:, None].repeat(1, n_sample_points).view(-1)), dim=0)
        #     self.env_ids = env_ids.tolist()
        #     self.mapping_goal_to_env_ids = # TODO !!!!

        # MPPI
        population = None
        resample_env_ids = None

        # Reset - Only needed for MPPI
        if torch.any(self.obs["resample_population"]):
            # get resample environments
            resample_env_ids = torch.where(self.obs["resample_population"])[0].tolist()
            self.optim.reset(resample_env_ids)
            # # MPPI
            # self.previous_solution[resample_env_ids] *= 0.0
            # population = self.previous_solution
        # else:
        #     population = self.previous_solution

        best_population, self.var = self.optim.optimize(
            obj_fun=self.func,
            env_ids=self.env_ids,
            x0=population,
            x0_env_ids=resample_env_ids,
            var0=None,
            callback=self.logging_callback,
        )
        # self.var is shape := (BS, TRAJ_LENGTH, CONTROL_DIM)
        # best_population is shape := (BS, TRAJ_LENGTH, CONTROL_DIM)

        # self.previous_solution = best_population

        if return_states:
            states = self.func(best_population[None], only_rollout=True)
            # states is shape := (BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM)
        else:
            states = None

        return states, best_population

    ###
    # FDM functions
    ###

    def b_obj_func_N_step(
        self,
        population: torch.Tensor,
        only_rollout: bool = False,
        control_mode: str | None = None,
        env_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Objective function called by optimizer.
        We dynamicially allocate everything given that the population can grow or shrink
        """
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]  # equal to the number of environments that have to be replanned
        TRAJ_LENGTH = population.shape[2]

        # override env_ids when given
        if env_ids is not None:
            self.env_ids = env_ids

        # start_state = self.get_start_state(BS, NR_TRAJ, env_ids=env_ids).clone()
        start_state = self.get_start_state(BS, NR_TRAJ).clone()

        if control_mode is None:
            control_mode = self.to_cfg.control

        if control_mode == "velocity_control":
            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

            # If the actions is small make the robot stand
            if self.to_cfg.set_actions_below_threshold_to_0:
                m_vel_lin = torch.norm(population[:, :, :, :2], p=2, dim=3) < self.to_cfg.vel_limit_lin
                m_vel_ang = torch.abs(population[:, :, :, 2]) < self.to_cfg.vel_limit_ang
                m_vel_lin = m_vel_lin[:, :, :, None].repeat(1, 1, 1, 3)
                m_vel_lin[:, :, :, 2] = False
                m_vel_ang = m_vel_ang[:, :, :, None].repeat(1, 1, 1, 3)
                m_vel_ang[:, :, :, :2] = False

                if m_vel_lin.sum() > 0:
                    population[m_vel_lin] = 0
                if m_vel_ang.sum() > 0:
                    population[m_vel_ang] = 0

            # Integrate the velocity actions to positions
            actions = population * self.to_cfg.dt

            # Cumsum is an inplace operation therefore the clone is necesasry
            cummulative_yaw = actions.clone()[:, :, :, -1].cumsum(2)

            # We need to take the non-linearity by the rotation into account
            r_vec1 = torch.stack([torch.cos(cummulative_yaw), torch.sin(cummulative_yaw)], dim=3)
            r_vec2 = torch.stack([-torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=3)

            so2 = torch.stack([r_vec1, r_vec2], dim=4)

            # Move the rotation in time and fill first timestep with identity - see math chapter
            so2 = torch.roll(so2, shifts=1, dims=2)
            so2[:, :, 0, :, :] = torch.eye(2, device=so2.device)[None, None].repeat(BS, NR_TRAJ, 1, 1)

            # TODO this may be not needed given that max velocity in the x,y should stay enforced
            # actions[:,:,:,:2] /= torch.norm(actions[:,:,:,:2],p=2, dim=3)[None] * 1.2

            actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ actions[:, :, :, :2].contiguous().reshape(
                -1, 2, 1
            )
            actions_local_frame = actions_local_frame.contiguous().reshape(BS, NR_TRAJ, TRAJ_LENGTH, 2)
            cumulative_position = (actions_local_frame).cumsum(dim=2)
            local_states = torch.cat([cumulative_position, cummulative_yaw[:, :, :, None]], dim=3)

            # Transform the states from the current base frame to the odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(local_states)  # this here should be from base to points -> se2_points_base

            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

        elif control_mode == "fdm" or control_mode == "fdm_baseline":

            # check if FDM model is provided
            assert self.fdm_model is not None, "FDM model is not set"

            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

            # get initial states
            if self.env_ids == slice(None):
                env_idx = torch.arange(BS)
            else:
                env_idx = self.env_ids if isinstance(self.env_ids, torch.Tensor) else torch.tensor(self.env_ids)

            # init final output buffers
            num_envs = len(env_idx)
            if control_mode == "fdm":
                local_states = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH, 4), device=self.fdm_model.device)
                energy_traj = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH, 1), device=self.fdm_model.device)
            else:
                local_states = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH, 2), device=self.fdm_model.device)
            if self.fdm_model.cfg.unified_failure_prediction:
                collision_prob_traj = torch.zeros((num_envs, NR_TRAJ), device=self.fdm_model.device)
            else:
                collision_prob_traj = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH), device=self.fdm_model.device)

            # process in mini-batches due to high memory requirements
            num_env_per_batch = math.ceil(max(self.to_cfg.batch_size / NR_TRAJ, 1))
            for mini_batch_idx in range(math.ceil(num_envs / num_env_per_batch)):
                curr_idx_range = [
                    num_env_per_batch * mini_batch_idx,
                    min(num_env_per_batch * (mini_batch_idx + 1), num_envs),
                ]
                curr_env_idx = env_idx[curr_idx_range[0] : curr_idx_range[1]]
                curr_batch_size = len(curr_env_idx)

                # get state history transformed into local frame
                state_history = state_history_transformer(
                    self.obs["states"],
                    curr_env_idx,
                    self.fdm_model.cfg.history_length,
                    self.fdm_model.cfg.exclude_state_idx_from_input,
                ).to(self.device)

                if control_mode == "fdm":
                    # Scale the extra observations
                    state_history[..., 5] = (state_history[..., 5] - self.fdm_model.hard_contact_obs_limits[0]) / (
                        self.fdm_model.hard_contact_obs_limits[1] - self.fdm_model.hard_contact_obs_limits[0]
                    )

                # make predictions
                # the population is BS, NR_TRAJ which is transformed to BS x NR_TRAJ
                # all other terms are repeated by the number of trajectories
                model_in = (
                    state_history.unsqueeze(1)
                    .repeat(1, NR_TRAJ, 1, 1)
                    .view(curr_batch_size * NR_TRAJ, state_history.shape[1], state_history.shape[2]),
                    (
                        self.obs["proprio_obs"][curr_env_idx]
                        .to(self.device)
                        .unsqueeze(1)
                        .repeat(1, NR_TRAJ, 1, 1)
                        .view(curr_batch_size * NR_TRAJ, *(self.obs["proprio_obs"].shape[1:]))
                    ),
                    (
                        self.obs["extero_obs"][curr_env_idx]
                        .type(torch.float32)
                        .to(self.device)
                        .unsqueeze(1)
                        .repeat(1, NR_TRAJ, *([1] * (self.obs["extero_obs"].dim() - 1)))
                        .view(curr_batch_size * NR_TRAJ, *(self.obs["extero_obs"].shape[1:]))
                        if "extero_obs" in self.obs
                        else torch.zeros(1)
                    ),
                    population[curr_idx_range[0] : curr_idx_range[1]].view(curr_batch_size * NR_TRAJ, TRAJ_LENGTH, -1),
                    (
                        self.obs["add_extero_obs"][curr_env_idx]
                        .type(torch.float32)
                        .to(self.device)
                        .unsqueeze(1)
                        .repeat(1, NR_TRAJ, *([1] * (self.obs["add_extero_obs"].dim() - 1)))
                        .view(curr_batch_size * NR_TRAJ, *(self.obs["add_extero_obs"].shape[1:]))
                        if "add_extero_obs" in self.obs
                        else torch.zeros(1)
                    ),
                )
                # make prediction
                with torch.no_grad():
                    if control_mode == "fdm":
                        curr_states, curr_collision_prob_traj, curr_energy_traj = self.fdm_model.forward(model_in)
                    else:
                        curr_states, curr_collision_prob_traj = self.fdm_model.forward(model_in)
                        if self.fdm_model.cfg.unified_failure_prediction:
                            curr_collision_prob_traj = torch.max(curr_collision_prob_traj, dim=-1)[0]

                # reshape states back to BS, NR_TRAJ
                local_states[curr_idx_range[0] : curr_idx_range[1]] = curr_states.view(
                    curr_batch_size, NR_TRAJ, *(curr_states.shape[1:])
                )
                collision_prob_traj[curr_idx_range[0] : curr_idx_range[1]] = curr_collision_prob_traj.view(
                    curr_batch_size, NR_TRAJ, *(curr_collision_prob_traj.shape[1:])
                )
                if control_mode == "fdm":
                    energy_traj[curr_idx_range[0] : curr_idx_range[1]] = curr_energy_traj.view(
                        curr_batch_size, NR_TRAJ, *(curr_energy_traj.shape[1:])
                    )

            if control_mode == "fdm":
                # transform the orientation encoding to a yaw angle
                local_states[:, :, :, 2] = torch.atan2(local_states[..., 2], local_states[..., 3])
                local_states = local_states[..., :3]
            else:
                # append a zero yaw angle to the states
                local_states = torch.cat([local_states, torch.zeros_like(local_states[..., 0])[..., None]], dim=-1)
            # transform states into odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(local_states)  # this here should be from base to points -> se2_points_base
            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

            # Integrate the velocity actions to positions for loss calculation
            actions = population * self.to_cfg.dt

        elif control_mode == "position_control":  # noqa: R506
            raise ValueError(
                "Not correctly implemented in the cost function handling the yaw actions forward, sidward motion"
                " correctly."
            )
            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM
            actions = population * self.to_cfg.dt
            states = start_state[:, :, None, :] + actions.cumsum(dim=2)

        else:
            raise ValueError(f"Control mode {control_mode} not supported")

        if only_rollout:
            return states

        # calculate the running cost
        running_cost = self.states_cost(states.clone(), actions)
        if self.to_cfg.debug:
            self.debug_info["states_running_cost"] = running_cost.clone()

        running_cost = running_cost.mean(dim=2)
        terminal_cost = self.terminal_cost(states[:, :, -1])

        total_cost = running_cost + terminal_cost

        if self.to_cfg.control == "fdm" or self.to_cfg.control == "fdm_baseline":
            collision_cost = self.collision_cost(states, collision_prob_traj)
            self.debug_info["collision_cost"] = collision_cost.clone()

            total_cost += collision_cost

        elif self.to_cfg.states_cost_w_cost_map:
            assert self.to_cfg.control != "fdm_baseline", "The height scan is not available for the baseline model"
            self.curr_cost_map_cost = self.cost_map_cost(local_states)
            self.debug_info["cost_map_cost"] = self.curr_cost_map_cost.clone()

            total_cost += self.curr_cost_map_cost

        if self.to_cfg.debug:
            self.debug_info["terminal_cost"] = terminal_cost.clone()
            self.debug_callback(states, total_cost)

        # make states and cost accessible for visualization
        self.states = states
        self.total_cost = total_cost
        self.population = population.permute(1, 0, 2, 3)

        # Return objective that needs to be maximized
        return -total_cost.T  # N_traj, BS

    def b_obj_func(
        self, population: torch.Tensor, only_rollout: bool = False, iteration: int | None = None
    ) -> torch.Tensor:
        """
        Objective function called by optimizer.
        """
        # We dynamicially allocate all these things given that the population can grow or shrink
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]
        TRAJ_LENGTH = population.shape[2]

        state = self.get_start_state(BS, NR_TRAJ)
        STATE_DIM = state.shape[-1]

        if not only_rollout:
            running_cost = torch.zeros((BS, NR_TRAJ), device=self.device)

        if self.to_cfg or only_rollout:
            # state sequence
            states = torch.zeros((BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM), device=self.device)

        action = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

        # with Timer(f"full rollout time {self.debug}, {only_rollout}"):
        for i in range(TRAJ_LENGTH):
            if not only_rollout:
                running_cost += self.states_cost(state[:, :, None], action[:, :, i, :][:, :, None])[:, :, 0]

            if self.to_cfg or only_rollout:
                states[:, :, i, :] = state.clone()

            state = self.forward_dynamics(state, action[:, :, i, :])

        if self.to_cfg.debug:
            self.debug_callback(states, running_cost)

        if only_rollout:
            return states

        return -((running_cost + self.terminal_cost(state)).T)

    ###
    # Cost functions
    ###

    def states_cost(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluates state cost of a sequence of states.

        Args:
            states (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM)): Sequence of states
            actions (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, ACTION_DIM)): Sequence of actions

        Returns:
            (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH)): Sequence of costs per state
        """
        # Compute action cost
        control_effort_trans_forward = torch.abs(actions[:, :, :, 0]) * self.to_cfg.state_cost_w_action_trans_forward
        control_effort_trans_side = torch.abs(actions[:, :, :, 1]) * self.to_cfg.state_cost_w_action_trans_side
        control_effort_rot = torch.abs(actions[:, :, :, 2]) * self.to_cfg.state_cost_w_action_rot

        # Compute early_goal_bonus:
        position_offset = torch.norm(states[:, :, :, :2] - self.obs["goal"][self.env_ids, None, None, :2], dim=3)
        # heading 0 if same; heading 1 if opposite
        goal_yaw = self.obs["goal"][self.env_ids, None, None, 2].repeat(1, states.shape[1], states.shape[2])
        heading = cosine_distance(states[:, :, :, 2], goal_yaw) / 2

        m = (position_offset < self.to_cfg.state_cost_early_goal_distance_offset) * (
            heading < self.to_cfg.state_cost_early_goal_heading_offset
        )

        # Velocity tracking
        velocity_tracking_cost = (
            torch.abs(torch.norm(actions[:, :, :, :2], p=2, dim=3) - self.to_cfg.state_cost_desired_velocity)
            * self.to_cfg.state_cost_velocity_tracking
        )

        # No action cost if we reached the goal
        control_effort_trans_forward[m] = 0
        control_effort_trans_side[m] = 0
        control_effort_rot[m] = 0
        velocity_tracking_cost[m] = 0

        # Early goal reaching reward
        early_goal_cost = -torch.ones_like(control_effort_trans_forward) * self.to_cfg.state_cost_w_early_goal_reaching
        early_goal_cost[~m] = 0

        # Early stopping reward
        # Minus Sign - maximize the percentage early stopping by minimizing the cost
        res = get_non_zero_action_length(actions)
        precentage_early_stopping = ((actions.shape[2] - (res + 1)) / actions.shape[2])[:, :, None].repeat(
            1, 1, actions.shape[2]
        )
        early_stopping_cost = -precentage_early_stopping * self.to_cfg.state_cost_w_early_stopping

        if self.to_cfg.debug:
            self.debug_info["states_control_effort_rot"] = control_effort_rot.clone()
            self.debug_info["states_control_effort_trans_forward"] = control_effort_trans_forward.clone()
            self.debug_info["states_control_effort_trans_side"] = control_effort_trans_side.clone()
            self.debug_info["states_early_goal_cost"] = early_goal_cost.clone()
            self.debug_info["states_early_stopping_cost"] = early_stopping_cost.clone()
            self.debug_info["velocity_tracking_cost"] = velocity_tracking_cost.clone()

        return (
            control_effort_trans_forward
            + control_effort_trans_side
            + control_effort_rot
            + early_goal_cost
            + early_stopping_cost
            + velocity_tracking_cost
        )

    def terminal_cost(self, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the terminal state cost for a given state.

        Args:
            state (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, STATE_DIM)): The terminal state to evaluate.

        Returns:
            torch.Tensor: The calculated terminal cost for the given state, with shape (BS, NR_TRAJ).
        """
        # compute and save position offset for visualization
        if self.to_cfg.pos_error_3d and self.terrain_analysis is not None:
            # get index on terrain analysis height map
            goal_pos_idx = (
                (
                    self.obs["goal"][self.env_ids, :2]
                    - torch.tensor(
                        [self.terrain_analysis.mesh_dimensions[2], self.terrain_analysis.mesh_dimensions[3]],
                        device=self.obs["goal"].device,
                    )
                )
                / self.terrain_analysis.cfg.grid_resolution
            ).int()
            goal_pos_idx[:, 0] = torch.clamp(goal_pos_idx[:, 0], 0, self.terrain_analysis.height_grid.shape[0] - 1)
            goal_pos_idx[:, 1] = torch.clamp(goal_pos_idx[:, 1], 0, self.terrain_analysis.height_grid.shape[1] - 1)

            final_path_pos_idx = (
                (
                    state[:, :, :2]
                    - torch.tensor(
                        [self.terrain_analysis.mesh_dimensions[2], self.terrain_analysis.mesh_dimensions[3]],
                        device=self.obs["goal"].device,
                    )
                )
                / self.terrain_analysis.cfg.grid_resolution
            ).int()
            final_path_pos_idx[..., 0] = torch.clamp(
                final_path_pos_idx[..., 0], 0, self.terrain_analysis.height_grid.shape[0] - 1
            )
            final_path_pos_idx[..., 1] = torch.clamp(
                final_path_pos_idx[..., 1], 0, self.terrain_analysis.height_grid.shape[1] - 1
            )
            # flatten final path pos idx
            final_path_pos_idx = final_path_pos_idx.view(-1, 2)

            z_height_goal = self.terrain_analysis.height_grid[goal_pos_idx[:, 0], goal_pos_idx[:, 1]]
            z_height_final = self.terrain_analysis.height_grid[final_path_pos_idx[:, 0], final_path_pos_idx[:, 1]]
            # back to original shape
            z_height_final = z_height_final.view(state.shape[0], state.shape[1])

            self.position_offset = torch.norm(
                torch.concatenate([state[:, :, :2], z_height_final[:, :, None]], dim=2)
                - torch.concatenate([self.obs["goal"][self.env_ids, None, :2], z_height_goal[:, None, None]], dim=2),
                dim=2,
            )

        else:
            self.position_offset = torch.norm(state[:, :, :2] - self.obs["goal"][self.env_ids, None, :2], dim=2)

        # heading_cossine_distance is 0 if the same
        # heading_cossine_distance is 2 if opposite vectors
        heading_cossine_distance = smallest_angle(
            state[:, :, 2], self.obs["goal"][self.env_ids, None, 2].repeat(1, state.shape[1])
        )
        # heading_reward -1 cost reduction if the same
        # heading_reward 0 cost reduction if opposite
        # heading_reward = heading_cossine_distance  # (-heading_cossine_distance / 2)-1
        # heading_reward[m] = 0

        if self.to_cfg.debug:
            self.debug_info["terminal_cost_position_offset"] = (
                self.position_offset.clone() * self.to_cfg.terminal_cost_w_position_error
            )
            self.debug_info["terminal_cost_heading_reward"] = (
                heading_cossine_distance.clone() * self.to_cfg.terminal_cost_w_rot_error
            )

        res = (
            self.position_offset * self.to_cfg.terminal_cost_w_position_error
            + heading_cossine_distance * self.to_cfg.terminal_cost_w_rot_error
        )

        if self.to_cfg.terminal_cost_use_threshold:
            m = self.position_offset < self.to_cfg.terminal_cost_distance_offset
            res[m] /= self.to_cfg.terminal_cost_close_reward

            if self.to_cfg.debug:
                self.debug_info["terminal_cost_total"] = res.clone()

        self.pose_cost = res.clone()

        return res

    def collision_cost(self, states: torch.Tensor, collision_traj: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the collision cost for a given estimated trajectory.

        Args:
            states: The estimated trajectory to evaluate shape (BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM).
            collision_traj: The collision probability to evaluate shape (BS, NR_TRAJ, TRAJ_LENGTH).

        Returns:
            torch.Tensor: The calculated collision probability cost for the given trajectory, with shape (BS, NR_TRAJ).
        """

        # penalize cost
        if self.fdm_model.cfg.unified_failure_prediction:
            cost = collision_traj * self.to_cfg.collision_cost_traj_factor
            cost[
                collision_traj > (self.fdm_model.cfg.collision_threshold - self.to_cfg.collision_cost_safety_factor)
            ] += self.to_cfg.collision_cost_high_risk_factor
        else:
            coll_idx = torch.any(
                collision_traj > (self.fdm_model.cfg.collision_threshold - self.to_cfg.collision_cost_safety_factor),
                dim=-1,
            )
            cost = torch.sum(collision_traj * self.to_cfg.collision_cost_traj_factor, dim=-1)
            cost[coll_idx] += self.to_cfg.collision_cost_high_risk_factor

        # get the distance between the trajectories
        num_envs, num_trajectories, T, _ = states.shape

        cost_pre = cost.clone()
        for env_id in range(num_envs):
            flattened_trajectories = states[env_id, :, :, :2].reshape(num_trajectories, -1)

            distance_matrix = cdist(
                flattened_trajectories.cpu().numpy(), flattened_trajectories.cpu().numpy(), metric="euclidean"
            )
            # Find the indices of the closest neighbors for each trajectory (excluding the trajectory itself)
            neighbors = np.argsort(distance_matrix, axis=1)[
                :, 1 : self.to_cfg.num_neighbors + 1
            ]  # Exclude the trajectory itself

            # Get the collision cost of the closest neighbors weighted by their distance
            distance_matrix = torch.tensor(distance_matrix, device=states.device)
            cost[env_id] += torch.sum(
                cost_pre[env_id][neighbors.flatten()].reshape(num_trajectories, self.to_cfg.num_neighbors)
                / distance_matrix[
                    torch.arange(distance_matrix.shape[0], device=states.device)[:, None].repeat(
                        1, self.to_cfg.num_neighbors
                    ),
                    neighbors,
                ],
                dim=-1,
            )

        self.collision_traj_cost = cost.clone()
        return cost

    def cost_map_cost(self, states: torch.Tensor) -> torch.Tensor:
        """Cost based on cost map generated from the height scan

        Args:
            states: States of the sampled trajectories in local frame

        Returns
            cost: Cost of the applied filters for every path
        """
        if self.env_ids == slice(None):
            env_idx = torch.arange(states.shape[0])
        else:
            env_idx = self.env_ids if isinstance(self.env_ids, torch.Tensor) else torch.tensor(self.env_ids)

        # handle the whole code batched due to memory limitations
        num_envs_per_batch = 20
        num_batches = math.ceil(len(env_idx) / num_envs_per_batch)

        cost = torch.zeros(states.shape[0], states.shape[1], device=states.device)

        for batch_idx in range(num_batches):
            curr_env_idx_range = torch.arange(
                num_envs_per_batch * batch_idx, min(num_envs_per_batch * (batch_idx + 1), len(env_idx))
            )
            curr_env_idx = env_idx[curr_env_idx_range]

            # get height-scan
            height_scan = self.obs["extero_obs"][curr_env_idx].squeeze(1).to(self.device)
            num_envs, grid_size_x, grid_size_y = height_scan.shape
            # get the tragversability map
            trav_map = torch.zeros_like(height_scan)
            trav_map[:, 3:-3, 3:-3] = self.traversability_filter(height_scan.to(torch.float32).unsqueeze(1)).squeeze(
                1
            )  # * 2
            # remove height scan from memory
            del height_scan

            if False:
                # NOTE: used to create the heuristic traversabilitty plot in the appendix of the paper
                import matplotlib.pyplot as plt

                all_idx = torch.arange(self.obs["extero_obs"].shape[0])
                curr_height_scan = self.obs["extero_obs"][all_idx].squeeze(1).to(self.device)
                curr_trav_map = self.traversability_filter(curr_height_scan.to(torch.float32).unsqueeze(1)).squeeze(
                    1
                )  # * 2

                # Ensure the save directory exists
                save_dir = os.path.join(FDM_DATA_DIR, "eval")
                os.makedirs(save_dir, exist_ok=True)

                # Number of columns per row
                cols_per_row = 6
                rows = math.ceil(self.obs["extero_obs"].shape[0] / cols_per_row)

                fig, axs = plt.subplots(
                    2 * rows,  # Multiply by 2 to accommodate two subplots (height scan and trav map) per column
                    cols_per_row,
                    figsize=(cols_per_row * 4, rows * 8),  # Adjust figsize based on cols_per_row and rows
                    gridspec_kw={"wspace": 0.4, "hspace": 0.6},  # Increase spacing between subplots
                )

                # Flatten axs for easier indexing (since it becomes 2D when multiple rows)
                axs = axs.reshape(2 * rows, cols_per_row)

                for i in range(self.obs["extero_obs"].shape[0]):
                    row, col = divmod(i, cols_per_row)
                    axs[2 * row, col].imshow(curr_height_scan[i].cpu().numpy())
                    axs[2 * row, col].axis("off")  # Optional: Hide axes for cleaner visuals
                    axs[2 * row + 1, col].imshow(curr_trav_map[i].cpu().numpy())
                    axs[2 * row + 1, col].axis("off")  # Optional: Hide axes for cleaner visuals
                    axs[2 * row, col].set_title(f"Height Scan {i}")
                    axs[2 * row + 1, col].set_title(f"Traversability Map {i}")

                plt.tight_layout()

                plt.savefig(os.path.join(save_dir, "trav_estimate_combined_figure.pdf"))
                plt.close()

                # Loop through each environment and create a combined figure
                for i in range(self.obs["extero_obs"].shape[0]):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots side by side

                    # Plot Height Scan
                    im1 = axes[0].imshow(curr_height_scan[i].cpu().numpy(), cmap="RdYlBu")
                    axes[0].set_title("Height Scan", fontsize=16)
                    axes[0].axis("off")
                    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar

                    # Plot Traversability Map
                    im2 = axes[1].imshow(curr_trav_map[i].cpu().numpy(), cmap="RdYlBu")
                    axes[1].set_title("Traversability Map", fontsize=16)
                    axes[1].axis("off")
                    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar

                    # Save the combined figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"trav_estimate_env{i}.pdf"))
                    plt.close(fig)

            # get the indexes of the points of the path on the height-map
            # NOTE: the height map is oriented with the robot x axis in the y direction of the height map and the robot y axis in the x direction of the height map
            #       therefore the x and y axis are swapped
            path_idx = torch.tensor(
                [[
                    grid_size_x / 2 - self.height_scan_offset[1] / self.height_scan_resolution,
                    grid_size_y / 2 - self.height_scan_offset[0] / self.height_scan_resolution,
                ]],
                device=states.device,
                dtype=torch.int32,
            ) + (states[curr_env_idx_range][..., [1, 0]] / self.height_scan_resolution).to(torch.int32) * torch.tensor(
                [-1, 1], device=states.device, dtype=torch.int32
            )

            # get risky coordinates for the robot
            # NOTE: torch.pi/2 has to be subtracted because we assume x going from left to right and y going from bottom to top and the robot shape assumes a different orientation
            alpha = states[curr_env_idx_range][:, :, :, 2, None].repeat(1, 1, 1, self.risky_xy.shape[0]) - torch.pi / 2
            cells_xy = self.risky_xy.clone()[None, None, None, :, :].repeat(len(curr_env_idx), *states.shape[1:3], 1, 1)

            so2 = torch.zeros(
                (len(curr_env_idx), *states.shape[1:3], self.risky_xy.shape[0], 2, 2), device=path_idx.device
            )
            so2[:, :, :, :, 0, 0] = torch.cos(alpha)
            so2[:, :, :, :, 1, 0] = torch.sin(alpha)
            so2[:, :, :, :, 0, 1] = -torch.sin(alpha)
            so2[:, :, :, :, 1, 1] = torch.cos(alpha)

            coordinates = torch.bmm(so2.reshape(-1, 2, 2), cells_xy.reshape(-1, 2)[:, :, None])[:, :, 0] + path_idx[
                :, :, :, None, :
            ].repeat(1, 1, 1, self.risky_xy.shape[0], 1).reshape(-1, 2)
            coordinates = coordinates.type(torch.long)

            # clip the idx to max indexes of the height map
            coordinates[:, 0] = torch.clamp(coordinates[:, 0], 0, grid_size_x - 1)
            coordinates[:, 1] = torch.clamp(coordinates[:, 1], 0, grid_size_y - 1)

            # Check all points of the robot shape
            env_idx_tensor = (
                torch.arange(len(curr_env_idx))[:, None]
                .repeat(1, states.shape[1] * states.shape[2] * self.risky_xy.shape[0])
                .reshape(-1)
            )
            filter_idx = trav_map[env_idx_tensor, coordinates[:, 0], coordinates[:, 1]] < 0.15
            path_filter = torch.any(filter_idx.reshape(num_envs, states.shape[1], -1), dim=-1)

            # filter paths
            curr_cost = torch.zeros(len(curr_env_idx), states.shape[1], device=states.device)
            curr_cost[path_filter] += self.to_cfg.state_cost_w_fatal_trav
            cost[curr_env_idx_range] += curr_cost

        if False:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].imshow(trav_map[0].cpu().numpy())
            path_map = torch.zeros_like(trav_map[0])
            coordinates_reshaped = coordinates.reshape(
                num_envs, states.shape[1] * states.shape[2] * self.risky_xy.shape[0], 2
            )[0]
            path_map[coordinates_reshaped[:, 0], coordinates_reshaped[:, 1]] = 1
            axs[1].imshow(path_map.cpu().numpy())
            cost_map = torch.zeros_like(trav_map[0])
            coordinates_traj = coordinates.reshape(
                num_envs, states.shape[1], states.shape[2] * self.risky_xy.shape[0], 2
            )[0]
            coordinates_filtered = coordinates_traj[path_filter[0]].reshape(-1, 2)
            cost_map[coordinates_filtered[:, 0], coordinates_filtered[:, 1]] = 1
            axs[2].imshow(cost_map.cpu().numpy())

        return cost

    def cost_map_cost_old(self, states: torch.Tensor, line_checking: bool = False) -> torch.Tensor:
        """Cost based on cost map generated from the height scan

        The cost-map is generated from the height-scan by applying the following heuristsics:
        - any increase over 0.3m is not traversable
        - TODO: going through non observable terrain

        .. note::
            NOT USED! Neural Network based cost map is used instead.

        Args:
            states: States of the sampled trajectories in local frame
            line_checking: If True the cost is calculated by checking the line segments of the path on the height map
                Problem: too slow. If False, just does a point wise check

        Returns
            cost: Cost of the applied filters for every path
        """
        with Timer("cost_map_cost: height map diff calculation"):
            # get height-scan
            height_scan = self.obs["extero_obs"][torch.arange(states.shape[0])].squeeze(1).to(self.device)
            num_envs, grid_size_x, grid_size_y = height_scan.shape
            # calc diff in both x and y
            height_diff = torch.diff(
                height_scan, dim=1, append=torch.zeros((num_envs, 1, grid_size_y), device=self.device)
            ) + torch.diff(height_scan, dim=2, append=torch.zeros((num_envs, grid_size_x, 1), device=self.device))
            height_diff = torch.abs(height_diff) > self.to_cfg.states_cost_w_cost_map_height_diff_thres

            # Define the dilation kernel based on robot size (assumes a square kernel for simplicity)
            robot_size = int(0.75 / self.height_scan_resolution)  # Adjust robot size here (kernel size)
            dilation_kernel = torch.ones((1, 1, robot_size, robot_size), device=self.device)

            # Add an extra dimension for batch compatibility with F.conv2d
            height_diff = height_diff.unsqueeze(1).to(torch.float32)

            # Perform dilation using 2D convolution, then convert back to boolean
            height_diff = F.conv2d(height_diff, dilation_kernel, padding=robot_size // 2) > 0
            height_diff = height_diff.squeeze(1).bool()  # Remove the extra dimension

            if False:
                import matplotlib.pyplot as plt

                plt.imshow(height_diff[0, 0].cpu().numpy())
                plt.savefig("/home/pascal/height_diff.png")
                plt.close()

        # get the indexes of the points of the path on the height-map
        # NOTE: the height map is oriented with the robot x axis in the y direction of the height map and the robot y axis in the x direction of the height map
        #       therefore the x and y axis are swapped
        path_idx = torch.tensor(
            [[
                grid_size_x / 2 - self.height_scan_offset[1] / self.height_scan_resolution,
                grid_size_y / 2 - self.height_scan_offset[0] / self.height_scan_resolution,
            ]],
            device=states.device,
            dtype=torch.int32,
        ) + (states[..., [1, 0]] / self.height_scan_resolution).to(torch.int32) * torch.tensor(
            [-1, 1], device=states.device, dtype=torch.int32
        )
        # clip the idx to max indexes of the height map
        path_idx[..., 0] = torch.clamp(path_idx[..., 0], 0, grid_size_x - 1)
        path_idx[..., 1] = torch.clamp(path_idx[..., 1], 0, grid_size_y - 1)

        if line_checking:
            # line segment filter actually covers the entire path but too slow
            with Timer("cost_map_cost: line segment checking"):
                path_idx_start = path_idx[:, :, :-1].reshape(num_envs, -1, 2)
                path_idx_end = path_idx[:, :, 1:].reshape(num_envs, -1, 2)

                filter_idx = torch.zeros(path_idx_start.shape[:2], dtype=torch.bool)

                for env_idx in range(num_envs):
                    for idx, (edge_start_idx, edge_end_idx) in enumerate(
                        zip(path_idx_start[env_idx], path_idx_end[env_idx])
                    ):
                        grid_idx_x, grid_idx_y = line(
                            edge_start_idx[0], edge_start_idx[1], edge_end_idx[0], edge_end_idx[1]
                        )

                        filter_idx[env_idx, idx] = torch.any(height_diff[env_idx][grid_idx_x, grid_idx_y])

                        # if env_idx == 0:
                        #     height_diff_plot = height_diff[env_idx].clone()
                        #     height_diff_plot[grid_idx_x, grid_idx_y] = 10
                        #     plt.imshow(height_diff_plot.cpu().numpy())
                        #     plt.savefig(f"/home/pascal/height_diff_traj_{idx}.png")
                        #     plt.close()

                # remove any path where parts of the way are going through a larger height diff
                path_filter = torch.any(filter_idx.reshape(num_envs, states.shape[1], -1), dim=-1)
        else:
            # POINT WISE filter (less precise but faster)
            path_idx_reshape = path_idx.reshape(-1, 2)
            env_idx = torch.arange(num_envs, device=states.device)[:, None, None].repeat(1, *states.shape[1:3])
            env_idx = env_idx.reshape(-1)
            filter_idx = height_diff[env_idx, path_idx_reshape[..., 0], path_idx_reshape[..., 1]]
            path_filter = torch.any(filter_idx.reshape(num_envs, states.shape[1], -1), dim=-1)

        # filter paths
        cost = torch.zeros(num_envs, states.shape[1], device=states.device)
        cost[path_filter] += self.to_cfg.state_cost_w_fatal_trav

        return cost

    ###
    # Helper functions
    ###

    def get_start_state(self, batch_size: int, nr_traj: int, env_ids: list[int] | None = None) -> torch.Tensor:
        """
        Initializes the start state for a batch of trajectories.

        Args:
            batch_size (int): The batch size.
            nr_traj (int): The number of trajectories.
            env_ids (list[int]): The environment ids for which to get the start state.

        Returns:
            torch.Tensor: The start state replicated for each trajectory, with shape (BS, NR_TRAJ, STATE_DIM).
        """
        if env_ids is None:
            env_ids = self.env_ids

        return self.obs["start"].clone()[env_ids, None, :].repeat(1, nr_traj, 1)

    def debug_callback(self, states, total_cost):
        b = 0
        best_traj = torch.argmin(total_cost[b])

        # Print statistics
        print("Running Cost:")
        for key in [
            "states_trav_cost",
            "states_control_effort_rot",
            "states_control_effort_trans_side",
            "states_control_effort_trans_forward",
            "states_early_goal_cost",
            "states_running_cost",
        ]:
            print(
                f"{key:>30}:    - Best Mean Total:"
                f" {round(self.debug_info[key][b, best_traj].mean().item(), 3):>10}       - Std Across Traj:"
                f" {round(self.debug_info[key][b].mean(dim=1).std().item(), 3):>10} "
            )
        #
        print("\nTerminal Cost:")
        for key in ["terminal_cost_position_offset", "terminal_cost_heading_reward", "terminal_cost"]:
            print(
                f"{key:>30}:    - Best Final: {round(self.debug_info[key][b, best_traj].item(), 3):>10}       - Std"
                f" Across Traj: {round(self.debug_info[key][b].std().item(), 3):>10} "
            )

    def logging_callback(self, population: torch.Tensor, values: torch.Tensor, iteration: int):
        if self.to_cfg.debug:
            min_v = values.min()
            print(f"Iteration: {iteration}, Values: {min_v}")

    def forward_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single sted forward dynamics model.
        """
        # Unit m + m/s * dt planning (actions are already integrated)
        return state + action
