# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import numpy as np
import torch
from scipy.spatial.distance import cdist

# necessary to prevent a bug in the CvBridge import see
# https://answers.ros.org/question/362388/cv_bridge_boost-raised-unreported-exception-when-importing-cv_bridge/
import cv2  # noqa: F401
import rospy
from cv_bridge import CvBridge
from pytictac import ClassTimer, accumulate_time

from fdm_navigation.cfg import CEMCfg  # noqa: F401
from fdm_navigation.cfg import MPPICfg  # noqa: F401
from fdm_navigation.cfg import ActionCfg, RobotCfg, TrajectoryOptimizerCfg, get_robot_shape
from fdm_navigation.helper import PathsToGridmap
from fdm_navigation.helper.math_utils import get_non_zero_action_length, get_se2, get_x_y_yaw, smallest_angle
from fdm_navigation.trajectory_optimizer import BatchedCEMOptimizer  # noqa: F401
from fdm_navigation.trajectory_optimizer import BatchedICEMOptimizer  # noqa: F401
from fdm_navigation.trajectory_optimizer import BatchedMPPIOptimizer  # noqa: F401
from fdm_navigation.visu import NumpyToRviz


class SimpleSE2TrajectoryOptimizer:
    def __init__(
        self,
        action_cfg: ActionCfg,
        robot_cfg: RobotCfg,
        optim,
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

        # Initialize Optimizer
        self.optim = optim

        self.cct = ClassTimer(objects=[self], names=["TrajOptim"], enabled=to_cfg.enable_timing)

        # Buffer for previous solution
        self.previous_solution = None
        self.var = None

        # high risk to stop before running into area with failure prediction
        self.high_risk_path = False

        # Set objective function
        if self.to_cfg.n_step_fwd:
            self.func = self.b_obj_func_N_step
        else:
            self.func = self.b_obj_func

        if self.to_cfg.init_debug:
            self.ntr = NumpyToRviz(
                init_node=False,
                cv_bridge=CvBridge(),
                image_topics=["image_raw_debug_cb"],
                gridmap_topics=["gridmap_debug_cb", "gridmap_paths_debug_cb"],
                pointcloud_topics=["pointcloud_debug_cb"],
                camera_info_topics=["camera_info_debug_cb"],
                marker_topics=["marker_debug_cb", "height_map_debug_cb"],
                path_topics=["path_debug_cb"],
                pose_topics=["goal_debug_cb", "start_debug_cb"],
            )

            self.debug_info = {}

    def set_fdm_classes(
        self,
        fdm_model: torch.nn.Module,
        height_map_center: torch.Tensor,
        height_map_resolution: float,
        height_map_size: tuple,
    ):
        self.fdm_model = fdm_model
        self.height_map_center = height_map_center.to(self.device)

        if self.to_cfg.init_debug:
            map_size = np.array(height_map_size)[::-1] / height_map_resolution
            map_size = list(np.round(map_size).astype(int))
            self.ptg_s = PathsToGridmap(map_size, height_map_resolution, self.device, "min_cost")
            self.height_map_resolution = height_map_resolution

    @accumulate_time
    def plan(self, obs: dict, robot_height: float = 0.0):
        """
        Initializes the observation dictionary with default values for planning.

        Args:
            obs (dict): A dictionary containing the following key-value pairs:
                - "goal": (torch.tensor, shape:=(BS, 3)): representing the goal with (x,y,yaw) in the odom frame.
                - "resample_population": bool: If the population should be resampled
                - "start": (torch.tensor, shape:=(BS, 3)): representing the start with (x,y,yaw)  in the odom frame.
                - "obs_proprio": torch.tensor: Proprioceptive observations for the FDM.
                - "obs_fdm_state": torch.tensor: State observations for the FDM.
                - "obs_extero": torch.tensor: Exteroceptive observations for the FDM.
        Returns:
            torch.Tensor: The planned trajectory with shape (BS, TRAJ_LENGTH, STATE_DIM).
            torch.Tensor: The planned velocity with shape (BS, TRAJ_LENGTH, CONTROL_DIM).

        """
        # save obs for later use
        self.obs = obs
        self.robot_height = robot_height

        # Reset - Only needed for MPPI
        if (self.previous_solution is None) or (self.obs["resample_population"]):
            self.optim.reset()
            # MPPI
            population = None

            # # TODO here one can also interpolate the heading
            # # TODO increase variance if goal is further away for sampling

            # start_state = self.get_start_state(1, 1).clone()
            # se2_odom_base = get_se2(start_state)  # Transformation from base to odom frame
            # se2_odom_goal = get_se2(self.obs["goal"])  # Transformation from goal to odom frame
            # se2_base_goal = torch.inverse(se2_odom_base) @ se2_odom_goal  # Transformation from base to goal frame
            # xyyaw_base_goal = get_x_y_yaw(se2_base_goal)

            # # Move omnidirectional and ignore heading.
            # dx = min(
            #     math.ceil(
            #         torch.abs(xyyaw_base_goal[0, 0, 0]) / (self.to_cfg.state_cost_desired_velocity * self.to_cfg.dt)
            #     ),
            #     self.action_cfg.traj_dim,
            # )
            # dy = min(
            #     math.ceil(
            #         torch.abs(xyyaw_base_goal[0, 0, 1]) / (self.to_cfg.state_cost_desired_velocity * self.to_cfg.dt)
            #     ),
            #     self.action_cfg.traj_dim,
            # )
            # x = torch.zeros((1, self.action_cfg.traj_dim, 1), device=self.device)
            # y = torch.zeros((1, self.action_cfg.traj_dim, 1), device=self.device)

            # x[:, :dx] = (xyyaw_base_goal[0, 0, 0]) / dx
            # y[:, :dy] = (xyyaw_base_goal[0, 0, 1]) / dy
            # z = torch.zeros((1, self.action_cfg.traj_dim, 1), device=self.device)

            # population = torch.cat([x, y, z], dim=2) / self.to_cfg.dt
        else:
            population = self.previous_solution

        best_population, self.var = self.optim.optimize(
            obj_fun=self.func,
            x0=population,
            var0=None,
            callback=self.logging_callback,
        )
        # self.var is shape := (BS, TRAJ_LENGTH, CONTROL_DIM)
        # best_population is shape := (BS, TRAJ_LENGTH, CONTROL_DIM)

        self.previous_solution = best_population
        states = self.func(best_population[None], only_rollout=True)
        # states is shape := (BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM)

        if self.to_cfg.control == "fdm" and self.to_cfg.collision_cost_high_risk_factor > 0.0:
            failure_risk = self.b_obj_func_N_step(best_population.unsqueeze(0), only_failure=True)
            if torch.any(failure_risk > self.to_cfg.collision_cost_threshold):
                rospy.logwarn("High Risk Trajectory")
                states = self.get_start_state(1, 1).clone()[:, :, None, :].repeat(1, 1, states.shape[2], 1)
                best_population = torch.zeros_like(best_population)
                self.high_risk_path = True
            else:
                self.high_risk_path = False

        if self.to_cfg.enable_timing:
            print(self.cct.__str__())

        return states, best_population

    @accumulate_time
    def b_obj_func_N_step(
        self,
        population: torch.Tensor,
        only_rollout: bool = False,
        only_failure: bool = False,
        only_rollout_states: bool = False,
        control_mode: str = None,
    ) -> torch.Tensor:
        """
        Objective function called by optimizer.
        We dynamicially allocate everything given that the population can grow or shrink
        """
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]
        TRAJ_LENGTH = population.shape[2]

        start_state = self.get_start_state(BS, NR_TRAJ).clone()

        if control_mode is None:
            control_mode = self.to_cfg.control

        if control_mode == "velocity_control":
            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

            # If the actions is small make the robot stand
            if self.to_cfg.set_actions_below_threshold_to_0:
                m_vel_lin = torch.norm(population[:, :, :, :2], p=2, dim=3) < self.to_cfg.vel_lin_min
                m_vel_ang = torch.abs(population[:, :, :, 2]) < self.to_cfg.vel_ang_min

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

            # TODO this may be not meeded given that max velocity in the x,y should stay enforced
            # actions[:,:,:,:2] /= torch.norm(actions[:,:,:,:2],p=2, dim=3)[None] * 1.2

            actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ actions[:, :, :, :2].contiguous().reshape(
                -1, 2, 1
            )
            actions_local_frame = actions_local_frame.contiguous().reshape(BS, NR_TRAJ, TRAJ_LENGTH, 2)
            cumulative_position = (actions_local_frame).cumsum(dim=2)
            states_base = torch.cat([cumulative_position, cummulative_yaw[:, :, :, None]], dim=3)

            # Transform the states_base from the current base frame to the odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(states_base)  # this here should be from base to points -> se2_points_base

            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

        elif control_mode == "fdm":

            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

            # make predictions
            # the population is BS, NR_TRAJ which is transformed to BS x NR_TRAJ
            # all other terms are repeated by the number of trajectories
            model_in = (
                self.obs["obs_fdm_state"].repeat(NR_TRAJ, 1, 1),
                self.obs["obs_proprio"].repeat(NR_TRAJ, 1, 1),
                self.obs["obs_extero"].repeat(NR_TRAJ, 1, 1, 1),
                population.squeeze(0),
                torch.zeros(1),
            )

            # make prediction
            with torch.no_grad():
                states_base, collision_prob_traj, energy_traj = self.fdm_model.forward(model_in)

            # transform the orientation encoding to a yaw angle
            states_base[..., 2] = torch.atan2(states_base[..., 2], states_base[..., 3])
            states_base = states_base[..., :3]

            # transform states_base into odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(states_base)  # this here should be from base to points -> se2_points_base
            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

            # Integrate the velocity actions to positions for loss calculation
            actions = population * self.to_cfg.dt

        elif control_mode == "position_control":
            raise ValueError(
                "Not correctly implemented in the cost function handling the yaw actions forward, sidward motion"
                " correctly."
            )
            # The population represents the normalized actions
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM
            actions = population * self.to_cfg.dt
            states = start_state[:, :, None, :] + actions.cumsum(dim=2)
        else:
            raise ValueError("Control mode not implemented")

        if only_rollout:
            return states
        elif only_failure:
            return collision_prob_traj
        elif only_rollout_states:
            return states, collision_prob_traj

        running_cost = self.states_cost(states.clone(), actions)
        if self.to_cfg.debug:
            self.debug_info["states_running_cost"] = running_cost.clone()

        running_cost = running_cost.mean(dim=2)
        terminal_cost = self.terminal_cost(states[:, :, -1])

        total_cost = running_cost + terminal_cost

        if control_mode == "fdm":
            collision_costs = self.collision_cost(states, collision_prob_traj)
            total_cost += collision_costs

            if self.to_cfg.debug:
                self.debug_info["collision_cost"] = collision_costs.clone()

        if self.to_cfg.debug:

            self.debug_info["terminal_cost"] = terminal_cost.clone()
            self.debug_callback(states, states_base, total_cost)

        # Return objective that needs to be maximized
        return -total_cost.T  # N_traj, BS

    @accumulate_time
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
        control_effort_rot = torch.abs(actions)[:, :, :, 2] * self.to_cfg.state_cost_w_action_rot

        # Compute early_goal_bonus:
        position_offset = torch.norm(states[:, :, :, :2] - self.obs["goal"][:, None, None, :2], dim=3)
        # heading 0 if same; heading 1 if opposite
        goal_yaw = self.obs["goal"][:, None, None, 2].repeat(1, states.shape[1], states.shape[2])
        heading_offset = smallest_angle(states[:, :, :, 2], goal_yaw)

        m = (position_offset < self.to_cfg.state_cost_early_goal_distance_offset) * (
            heading_offset < self.to_cfg.state_cost_early_goal_heading_offset
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
        precentage_early_stopping = ((states.shape[2] - (res + 1)) / states.shape[2])[:, :, None].repeat(
            1, 1, states.shape[2]
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

    @accumulate_time
    def terminal_cost(self, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the terminal state cost for a given state.

        Args:
            state (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, STATE_DIM)): The terminal state to evaluate.

        Returns:
            torch.Tensor: The calculated terminal cost for the given state, with shape (BS, NR_TRAJ).
        """

        position_offset = torch.norm(state[:, :, :2] - self.obs["goal"][:, None, :2], dim=2)

        # heading_cossine_distance is 0 if the same
        # heading_cossine_distance is 2 if opposite vectors
        heading_offset = smallest_angle(state[:, :, 2], self.obs["goal"][:, None, 2].repeat(1, state.shape[1]))
        # heading_reward -1 cost reduction if the same
        # heading_reward 0 cost reduction if opposite
        # heading_reward = heading_cossine_distance  # (-heading_cossine_distance / 2)-1
        # heading_reward[m] = 0

        if self.to_cfg.debug:
            self.debug_info["terminal_cost_position_offset"] = (
                position_offset.clone() * self.to_cfg.terminal_cost_w_position_error
            )
            self.debug_info["terminal_cost_heading_offset"] = (
                heading_offset.clone() * self.to_cfg.terminal_cost_w_rot_error
            )

        res = (
            position_offset * self.to_cfg.terminal_cost_w_position_error
            + heading_offset * self.to_cfg.terminal_cost_w_rot_error
        )

        if self.to_cfg.terminal_cost_use_threshold:
            m = position_offset < self.to_cfg.terminal_cost_distance_offset
            res[m] /= self.to_cfg.terminal_cost_close_reward

            if self.to_cfg.debug:
                self.debug_info["terminal_cost_total"] = res.clone()

        return res

    def collision_cost(self, states: torch.Tensor, collision_traj: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the collision cost for a given estimated trajectory.

        Args:
            states: The estimated trajectory to evaluate shape (BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM).
            collision_traj (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH)): The collision probability to evaluate.

        Returns:
            torch.Tensor: The calculated collision probability cost for the given trajectory, with shape (BS, NR_TRAJ).
        """

        # penalize cost
        if len(collision_traj.shape) == 1:
            cost = collision_traj * self.to_cfg.collision_cost_traj_factor
            cost[collision_traj > self.to_cfg.collision_cost_threshold] += self.to_cfg.collision_cost_high_risk_factor
        else:
            cost = torch.sum(collision_traj * self.to_cfg.collision_cost_traj_factor, dim=-1)
            cost[
                torch.any(collision_traj > self.to_cfg.collision_cost_threshold, dim=-1)
            ] += self.to_cfg.collision_cost_high_risk_factor

        num_neighbors = 2

        # get the distance between the trajectories
        _, num_trajectories, T, _ = states.shape

        cost_pre = cost.clone()
        flattened_trajectories = states[0, :, :, :2].reshape(num_trajectories, -1)

        distance_matrix = cdist(
            flattened_trajectories.cpu().numpy(), flattened_trajectories.cpu().numpy(), metric="euclidean"
        )
        # Find the indices of the closest neighbors for each trajectory (excluding the trajectory itself)
        neighbors = np.argsort(distance_matrix, axis=1)[:, 1 : num_neighbors + 1]  # Exclude the trajectory itself

        # Get the collision cost of the closest neighbors weighted by their distance
        distance_matrix = torch.tensor(distance_matrix, device=states.device)
        cost += torch.sum(
            cost_pre[neighbors.flatten()].reshape(num_trajectories, num_neighbors)
            / distance_matrix[
                torch.arange(distance_matrix.shape[0], device=states.device)[:, None].repeat(1, num_neighbors),
                neighbors,
            ],
            dim=-1,
        )

        self.collision_traj_cost = cost.clone()
        return cost

    """
    Helper functions
    """

    @accumulate_time
    def get_start_state(self, batch_size: int, nr_traj: int) -> torch.Tensor:
        """
        Initializes the start state for a batch of trajectories.

        Args:
            batch_size (int): The batch size.
            nr_traj (int): The number of trajectories.

        Returns:
            torch.Tensor: The start state replicated for each trajectory, with shape (BS, NR_TRAJ, STATE_DIM).
        """
        return self.obs["start"].clone()[:, None, :].repeat(1, nr_traj, 1)

    def debug_callback(self, states, states_base, total_cost):
        """
        Draw debug features

        Args:
            states: The estimated trajectory in odom frame
            states_base: The estimated trajectory in base frame
            total_cost: The total cost of the trajectory

        """

        try:
            b = 0
            best_traj = torch.argmin(total_cost[b])

            # Print statistics
            print("Running Cost:")
            for key in self.debug_info.keys():
                with contextlib.suppress(Exception):
                    print(
                        f"{key:>40}:    - Best Mean Total:"
                        f" {round(self.debug_info[key][b, best_traj].mean().item(), 3):>10}       - Std Across Traj:"
                        f" {round(self.debug_info[key][b].mean(dim=1).std().item(), 3):>10} "
                    )

            print("\nTerminal Cost:")
            for key in ["terminal_cost_position_offset", "terminal_cost_heading_offset", "terminal_cost"]:
                print(
                    f"{key:>40}:    - Best Final: {round(self.debug_info[key][b, best_traj].item(), 3):>10}       - Std"
                    f" Across Traj: {round(self.debug_info[key][b].std().item(), 3):>10} "
                )

            # Visualize all the trajectories
            linesegments = states[b].clone().reshape(-1, 3)
            linesegments[:, 2] = self.robot_height
            self.ntr.marker(
                "marker_debug_cb", linesegments.cpu().numpy(), reference_frame=self.frame_id, color=(0.2, 0.2, 0.2, 0.5)
            )
            self.ntr.path("path_debug_cb", path=None, xy_yaw=states[b, best_traj, :, :], reference_frame=self.frame_id)

            # Publish the goal and start pose
            goal = self.obs["goal"].clone()[b]
            goal[2] = self.robot_height
            # TODO add correct heading
            self.ntr.pose("goal_debug_cb", position=goal, orientation=[0, 0, 0, 1], reference_frame=self.frame_id)
            start = self.obs["start"].clone()[b]
            start[2] = self.robot_height
            # TODO add correct heading
            self.ntr.pose("start_debug_cb", position=start, orientation=[0, 0, 0, 1], reference_frame=self.frame_id)

            # Pre-process path cost for visualization
            c = total_cost[0, :].clone()
            c -= c.min()
            c = torch.log(c)
            c /= c.max()

            # Project paths on gridmap
            paths_on_gridmap_s = self.ptg_s(
                # states_base[b, :, :, :2].clone(),
                states_base[..., [1, 0]].clone(),
                c.clone(),
                self.height_map_center[[1, 0]],
            )

            m_invalid = torch.logical_or(torch.isnan(paths_on_gridmap_s), torch.isinf(paths_on_gridmap_s))

            if (~m_invalid).sum() == 0:
                fill_value = 1
            else:
                fill_value = paths_on_gridmap_s[~m_invalid].max()
                paths_on_gridmap_s[m_invalid] = fill_value + 0.2

            # NOTE: transpose necessary as x and y is swapped in the visualization function
            gridmap = torch.stack(
                [paths_on_gridmap_s.T],
                dim=0,
            )
            self.ntr.gridmap_from_numpy(
                "gridmap_debug_cb",
                gridmap.cpu().numpy(),
                resolution=self.height_map_resolution,
                layers=["paths_on_gridmap"],
                reference_frame="base",  # self.frame_id,
                x=(self.ptg_s.gridmap_shape[1] * self.height_map_resolution / 2 - self.height_map_center[0]),
            )

        except Exception as e:
            print("Debug callback failed with error: ", e)
            print("If the input resolution is not 200 x 200 the path projection on the GridMap fails.")

    def logging_callback(self, population: torch.Tensor, values: torch.Tensor, iteration: int):
        if self.to_cfg.debug:
            min_v = values.min()
            print(f"Iteration: {iteration}, Values: {min_v}")

    def b_obj_func(self, population: torch.Tensor, only_rollout: bool = False) -> torch.Tensor:
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

        if self.self.to_cfg or only_rollout:
            # state sequence
            states = torch.zeros((BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM), device=self.device)

        action = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

        for i in range(TRAJ_LENGTH):
            if not only_rollout:
                running_cost += self.states_cost(state[:, :, None], action[:, :, i, :][:, :, None])[:, :, 0]

            if self.self.to_cfg or only_rollout:
                states[:, :, i, :] = state.clone()

            state = self.forward_dynamics(state, action[:, :, i, :])

        if self.to_cfg.debug:
            self.debug_callback(states, running_cost)

        if only_rollout:
            return states

        return -((running_cost + self.terminal_cost(state)).T)

    def forward_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single sted forward dynamics model.
        """
        # Unit m + m/s * dt planning (actions are already integrated)
        return state + action
