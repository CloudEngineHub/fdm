# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import prettytable
import torch

import cv2
import hydra
import isaaclab.utils.math as math_utils
import omegaconf
import pypose as pp
import seaborn as sns
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from nav_collectors.terrain_analysis import TerrainAnalysis

from fdm.env_cfg.ui import PlannerEnvWindow
from fdm.mdp import GoalCommand

from ..model import FDMModel, FDMModelVelocityMultiStep, FDMProprioceptionModel, FDMProprioceptionVelocityModel
from .planner_cfg import FDMPlannerCfg
from .sampling_planner import SimpleSE2TrajectoryOptimizer

# can only be imported if gui activated
try:
    from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
except ImportError:
    omni_debug_draw = None


class FDMPlanner:
    def __init__(self, cfg: FDMPlannerCfg, planner_cfg: dict, args_cli):
        self.cfg = cfg
        self.planner_cfg = planner_cfg
        self.args_cli = args_cli

        # set drawing parameters
        self.nb_draw_traj = 10
        self.step_draw_traj = 2

        # update cfg
        self.cfg.env_cfg.scene.num_envs = self.args_cli.num_envs

        # init debug draw
        if omni_debug_draw:
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

            # Generate the color palette
            colors = sns.color_palette("RdYlBu", 10)
            self.cost_population_colors = [(r, g, b, 0.5) for r, g, b in reversed(colors)]
            self.coll_population_colors = [self.cost_population_colors[0], self.cost_population_colors[-1]]
            # init debug markers for selected velocity commands (only for visual evaluation)
            self.command_vel_visualizer: list[VisualizationMarkers] = []
            for traj_idx in range(self.planner_cfg["traj_dim"]):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_command"
                marker_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)
                self.command_vel_visualizer.append(VisualizationMarkers(marker_cfg))
                self.command_vel_visualizer[traj_idx].set_visibility(True)
        else:
            self.draw_interface = None

        # override the resampling command of the command generator with `trainer_cfg.command_timestep`
        self.cfg.env_cfg.episode_length_s = self.cfg.max_path_time

        # configure the decimation of the planner based on the intended frequency
        # that means the amount of simulation steps with internal low-level decimation have to executed before a new
        # planner prediction is made
        self.planner_decimation = int(
            (1 / self.cfg.frequency)
            / (self.cfg.env_cfg.sim.dt * self.cfg.env_cfg.actions.velocity_cmd.low_level_decimation)
        )

        # setup
        self.setup()

        # resume model and set into eval mode
        self.log_root_path = os.path.join("logs", "fdm", self.cfg.experiment_name)
        self.log_root_path = os.path.abspath(self.log_root_path)
        resume_path = get_checkpoint_path(self.log_root_path, self.cfg.load_run, "model_collection_round_01.pth")
        self.model.load(resume_path)
        self.model.eval()
        print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

    """
    Properties
    """

    @property
    def device(self) -> str:
        """The device to use for training."""
        return self.env.device

    """
    Operations
    """

    def setup(self):
        # setup environment
        self.env: ManagerBasedRLEnv = ManagerBasedRLEnv(self.cfg.env_cfg)
        # setup model
        self.model: FDMModel | FDMModelVelocityMultiStep | FDMProprioceptionModel | FDMProprioceptionVelocityModel = (
            self.cfg.model_cfg.class_type(cfg=self.cfg.model_cfg, device=self.device)
        )
        self.model.to(self.device)
        # setup planner
        cfg = omegaconf.OmegaConf.create(self.planner_cfg)
        self.planner: SimpleSE2TrajectoryOptimizer = hydra.utils.instantiate(cfg.to)
        self.planner.set_fdm_classes(fdm_model=self.model, env=self.env)

        # buffers
        # -- buffer for visualization of all states of the population and the cost of each prediction
        self.population_states = torch.zeros(
            self.env.num_envs,
            self.planner.optim.population_size,
            self.cfg.model_cfg.prediction_horizon,
            3,
            device=self.device,
        )
        self.population_costs = torch.zeros(self.env.num_envs, self.planner.optim.population_size, device=self.device)
        self.population_goal_offset = torch.zeros(
            self.env.num_envs, self.planner.optim.population_size, device=self.device
        )
        self.population_goal_reward = torch.zeros(
            self.env.num_envs, self.planner.optim.population_size, device=self.device
        )
        self.population_collision_costs = torch.zeros(
            self.env.num_envs, self.planner.optim.population_size, device=self.device
        )
        self.population_perfect_velocity = torch.zeros(
            self.env.num_envs,
            self.planner.optim.population_size,
            self.cfg.model_cfg.prediction_horizon,
            3,
            device=self.device,
        )
        # -- get the feet index in the contact sensor
        self.feet_idx, _ = self.env.scene.sensors["contact_forces"].find_bodies(".*FOOT")
        self.feet_contact = torch.zeros(
            (self.env.num_envs, len(self.feet_idx)), dtype=torch.bool, device=self.env.device
        )
        # -- env_step counter to decide when to do a new prediction
        self.decimation_counter = torch.zeros(self.env.num_envs, device=self.env.device)
        # -- count steps where env is not moving to resample the population
        self._non_moving_step_counter = torch.zeros(self.env.num_envs, device=self.env.device)

        # setup observation buffers
        self._init_obs_buffers()

        print("[INFO]: Setup complete.")

    def navigate(self, applied_planner: str = "mppi_fdm", debug: bool = False) -> dict:
        """Run the visual evaluation of the model.

        Args:
            initial_warm_up (bool, optional): Let the environments run until the history buffers are filled. Then
                performs a first prediciton and return the results. Defaults to True.

        """
        # reset the environment
        self._reset_obs_buffers(torch.arange(self.env.num_envs, device=self.env.device))
        with torch.inference_mode():
            obs, _ = self.env.reset(1)
        # execute dummy step to get the initial state
        with torch.inference_mode():
            obs, _, _, _, _ = self.env.step(torch.zeros(self.env.num_envs, 3, device=self.env.device))

        # get initial actions
        # first small random actions until the obs buffer is filled
        initial_rand_actions = (
            torch.randn(self.env.num_envs, self.cfg.model_cfg.prediction_horizon, 3, device=self.env.device) * 0.05
        )
        se2_velocity_b = initial_rand_actions.clone()
        se2_positions_w = torch.zeros(
            self.env.num_envs, self.cfg.model_cfg.prediction_horizon, 3, device=self.env.device
        )

        # extract goal command generator
        goal_command: GoalCommand = self.env.command_manager._terms["command"]

        # path length and episode time are getting overwritten when reset, save the last one to catch the correct value
        # for the terminated environments
        traversed_path_length_env_temp = goal_command.path_length_command.clone()
        episode_length_buf_temp = self.env.episode_length_buf.clone()

        # evaluation buffer
        goal_success = torch.zeros(goal_command.nb_generated_paths, dtype=torch.bool, device=self.env.device)
        traversed_path_length = torch.zeros(goal_command.nb_generated_paths, device=self.env.device)
        trajectory_time = torch.zeros(goal_command.nb_generated_paths, device=self.env.device)
        path_rrt_length = torch.zeros(goal_command.nb_generated_paths, device=self.env.device)
        robot_trajectory_length = torch.zeros(self.env.num_envs, device=self.env.device)
        robot_past_position = self.env.scene.articulations["robot"].data.root_pos_w.clone()
        finished_path_counter = 0
        debug_path_counter = 0

        while not goal_command.all_path_completed:
            # step environment
            with torch.inference_mode():
                # NOTE: goal reached is the only source for an time-out
                obs, _, dones, goal_reached, _ = self.env.step(se2_velocity_b[:, 0, :])
            # NOTE: technically does not collide when the goal is reached, still have to resample
            env_reset = torch.logical_or(dones, goal_reached)

            ###
            # Determine feet contact
            ###
            # reset for terminated envs
            self.feet_contact[env_reset] = False
            # Note: only start recording and changing actions when all feet have touched the ground
            self.feet_contact[
                torch.norm(self.env.scene.sensors["contact_forces"].data.net_forces_w[:, self.feet_idx], dim=-1) > 1
            ] = True
            feet_all_contact = torch.all(self.feet_contact, dim=-1)

            ###
            # update obs buffer
            ###
            self._reset_obs_buffers(env_reset)
            self._update_obs_buffers(
                state=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                feet_contact=feet_all_contact,
            )

            ###
            # Get the actions
            ###
            # for reset environments, initialize the actions with small values, they will be reset when the planner
            # decimation is reached for the first time. Don't directly replan, as the reset_buffer is not filled
            if torch.any(env_reset):
                se2_velocity_b[env_reset] = initial_rand_actions[env_reset]
                se2_positions_w[env_reset] = torch.zeros(
                    env_reset.sum().item(), self.cfg.model_cfg.prediction_horizon, 3, device=self.device
                )
                self.decimation_counter[env_reset] = 0

            # get the environments where to replan
            replan_envs = (
                torch.logical_and(
                    (self.decimation_counter == self.planner_decimation),
                    self._obs_env_step_counter >= self.model.cfg.prediction_horizon,
                )
                .nonzero()
                .squeeze(1)
                .tolist()
            )
            # get the environments where the buffer is not filled yet
            replan_not_filled = (
                torch.logical_and(
                    (self.decimation_counter == self.planner_decimation),
                    self._obs_env_step_counter < self.model.cfg.prediction_horizon,
                )
                .nonzero()
                .squeeze(1)
                .tolist()
            )

            # resample when the buffer filled for the first time
            obs["planner_obs"]["resample_population"] = self._obs_env_step_counter == self.model.cfg.prediction_horizon

            # se2_positions_w is return with shape (num_envs, 1, trajectory_length, state_dim) where state_dim is 3
            # se2_velocity_b is return with shape (num_envs, trajectory_length, action_dim) where action_dim is 3
            # obs["planner_obs"]["resample_population"] = env_reset

            # also resample the population for standing environments
            non_moving_samples = self._non_moving_step_counter > self.cfg.movement_resample_count
            obs["planner_obs"]["resample_population"] |= non_moving_samples
            if torch.any(non_moving_samples) and debug:
                print(
                    f"[DEBUG]: Resampling population of env {torch.where(non_moving_samples)[0]} bc non-moving state."
                )

            # planning
            if len(replan_envs) > 0:
                # if torch.any(obs["planner_obs"]["resample_population"]):
                #     print(f"[INFO]: Resampling population for {torch.where(obs['planner_obs']['resample_population'])[0].tolist()}")
                #     print(f"[INFO]: Buffer step idx {self._obs_env_step_counter[obs['planner_obs']['resample_population'].cpu()].tolist()}")

                # add states, proprio and extero observations to the planner obs
                obs["planner_obs"]["states"] = self._state_history.clone()
                obs["planner_obs"]["proprio_obs"] = self._proprio_obs_history.clone()
                obs["planner_obs"]["extero_obs"] = obs["fdm_obs_exteroceptive"].clone()

                se2_positions_w_pred, se2_velocity_b[replan_envs] = self.planner.plan(
                    obs=obs["planner_obs"], env_ids=replan_envs
                )
                se2_positions_w[replan_envs] = se2_positions_w_pred[:, 0, :, :]

            if len(replan_not_filled) > 0:
                # print(f"[INFO]: Resampling population for not filled buffer {replan_not_filled}")
                # print(f"[INFO]: Buffer step idx {self._obs_env_step_counter[replan_not_filled].tolist()}")

                # replace actions if obs buffer has not been filled yet
                se2_velocity_b[replan_not_filled] = torch.roll(initial_rand_actions[replan_not_filled], 1, dims=1)
                se2_positions_w[replan_not_filled] *= 0

            # update decimation counter
            self.decimation_counter[replan_envs] = 0
            self.decimation_counter[replan_not_filled] = 0
            self.decimation_counter += 1
            # check if applied command is lower than threshold
            self._non_moving_step_counter += torch.norm(se2_velocity_b[:, 0, :], dim=-1) < self.cfg.movement_threshold
            self._non_moving_step_counter[
                torch.norm(se2_velocity_b[:, 0, :], dim=-1) >= self.cfg.movement_threshold
            ] = 0

            ###########################################################################################################
            # Planner EVALUATION
            ###########################################################################################################

            # make sure done and goal_reached environments are still updated
            done_updated = dones & ~goal_command.prev_not_updated_envs
            goal_reached_updated = goal_reached & ~goal_command.prev_not_updated_envs
            # record evaluation metrics for done environments
            goal_success[finished_path_counter : finished_path_counter + done_updated.sum()] = False
            trajectory_time[finished_path_counter : finished_path_counter + done_updated.sum()] = (
                episode_length_buf_temp[done_updated]
            )
            traversed_path_length[finished_path_counter : finished_path_counter + done_updated.sum()] = (
                robot_trajectory_length[done_updated]
            )
            path_rrt_length[finished_path_counter : finished_path_counter + done_updated.sum()] = (
                traversed_path_length_env_temp[done_updated]
            )
            finished_path_counter += int(done_updated.sum())
            # record evaluation metrics for time-out = goal reached environments
            goal_success[finished_path_counter : finished_path_counter + goal_reached_updated.sum()] = True
            trajectory_time[finished_path_counter : finished_path_counter + goal_reached_updated.sum()] = (
                episode_length_buf_temp[goal_reached_updated]
            )
            traversed_path_length[finished_path_counter : finished_path_counter + goal_reached_updated.sum()] = (
                robot_trajectory_length[goal_reached_updated]
            )
            path_rrt_length[finished_path_counter : finished_path_counter + goal_reached_updated.sum()] = (
                traversed_path_length_env_temp[goal_reached_updated]
            )
            finished_path_counter += int(goal_reached_updated.sum())
            # update robot specific trajectory lentg buffer
            robot_trajectory_length += torch.norm(
                self.env.scene.articulations["robot"].data.root_pos_w - robot_past_position, dim=-1
            )
            robot_past_position = self.env.scene.articulations["robot"].data.root_pos_w.clone()
            robot_trajectory_length[dones] = 0
            robot_trajectory_length[goal_reached] = 0
            # overwrite the command trajectory length for the terminated environments
            traversed_path_length_env_temp[dones] = goal_command.path_length_command[dones].clone()
            traversed_path_length_env_temp[goal_reached] = goal_command.path_length_command[goal_reached].clone()
            # store last episode_lenght
            episode_length_buf_temp = self.env.episode_length_buf.clone()

            ###
            # Visualize the selected path
            ###
            if self.draw_interface is not None:
                self._visualize(replan_envs, se2_positions_w, se2_velocity_b, obs)

            ###
            # Print eval info
            ###
            if torch.any(done_updated):
                print(f"[INFO]: Done updated environments: {torch.where(done_updated)[0].tolist()}")
            if torch.any(goal_reached_updated):
                print(f"[INFO]: Goal reached updated environments: {torch.where(goal_reached_updated)[0].tolist()}")

            debug_path_counter += done_updated.sum() + goal_reached_updated.sum()
            # for every 50 paths, given current stats
            if debug_path_counter > 50:
                debug_path_counter = 0
                table = prettytable.PrettyTable()
                table.field_names = ["Metric", "Success", "Fail", "All"]
                for key, value in self._planner_eval_metrics(
                    finished_path_counter, path_rrt_length, traversed_path_length, goal_success, trajectory_time
                ).items():
                    table.add_row([key, value["Success"], value["Fail"], value["All"]])
                print("[INFO] Planner Evaluation\n", table)

        # get final eval metrics
        metrics = self._planner_eval_metrics(
            finished_path_counter, path_rrt_length, traversed_path_length, goal_success, trajectory_time
        )

        # print final evaluation
        self.print_metrics(metrics)

        # dump results into yaml
        dump_yaml(self.log_root_path + f"/planner_eval_metric_{applied_planner}.yaml", metrics)

        return metrics

    def test(self, cameras: list | None = None):
        """Test Planner in Demo environments with fix start and goal positions."""
        # set manual seed
        torch.manual_seed(0)
        # reset the environment
        with torch.inference_mode():
            obs, _ = self.env.reset(0)
        # execute dummy step to get the initial state
        with torch.inference_mode():
            obs, _, _, _, _ = self.env.step(torch.zeros(self.env.num_envs, 3, device=self.env.device))

        # get initial actions
        # first small random actions until the obs buffer is filled
        torch.manual_seed(0)
        initial_rand_actions = (
            torch.randn(self.env.num_envs, self.cfg.model_cfg.prediction_horizon, 3, device=self.env.device) * 0.1
        )
        se2_velocity_b = initial_rand_actions.clone()
        se2_positions_w = torch.zeros(
            self.env.num_envs, self.cfg.model_cfg.prediction_horizon, 3, device=self.env.device
        )

        # extra imports when images should be recorded and define save paths
        if cameras is not None:
            # setup image save path
            resume_path = get_checkpoint_path(self.log_root_path, self.cfg.load_run, self.cfg.load_checkpoint)
            directory_path = os.path.dirname(resume_path)
            render_path = os.path.join(directory_path, "Planning_render")
            os.makedirs(render_path, exist_ok=True)
            if self.env._window.current_cost_viz_mode is not None:
                suffix = f"_{self.env._window.current_cost_viz_mode}"
            else:
                suffix = ""
            cam_save_path = []
            for idx, camera in enumerate(cameras):
                cam_save_path.append(os.path.join(render_path, f"camera_{self.args_cli.env}_{idx}{suffix}"))
                os.makedirs(cam_save_path[-1], exist_ok=True)

            # stop recording when goal is reached
            goal_ever_reached = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

            # counter for rendering
            render_counter = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.int)

        while True:
            # step environment
            with torch.inference_mode():
                # NOTE: goal reached is the only source for an time-out
                obs, _, dones, goal_reached, _ = self.env.step(se2_velocity_b[:, 0, :])
            # NOTE: technically does not collide when the goal is reached, still have to resample
            env_reset = torch.logical_or(dones, goal_reached)

            ###
            # Determine feet contact
            ###
            # reset for terminated envs
            self.feet_contact[env_reset] = False
            # Note: only start recording and changing actions when all feet have touched the ground
            self.feet_contact[
                torch.norm(self.env.scene.sensors["contact_forces"].data.net_forces_w[:, self.feet_idx], dim=-1) > 1
            ] = True
            feet_all_contact = torch.all(self.feet_contact, dim=-1)

            ###
            # update obs buffer
            ###
            self._reset_obs_buffers(env_reset)
            self._update_obs_buffers(
                state=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                feet_contact=feet_all_contact,
            )

            ###
            # Get the actions
            ###
            # for reset environments, initialize the actions with small values, they will be reset when the planner
            # decimation is reached for the first time. Don't directly replan, as the reset_buffer is not filled
            if torch.any(env_reset):
                se2_velocity_b[env_reset] = initial_rand_actions[env_reset]
                se2_positions_w[env_reset] = torch.zeros(
                    env_reset.sum().item(), self.cfg.model_cfg.prediction_horizon, 3, device=self.device
                )
                self.decimation_counter[env_reset] = 0

            # get the environments where to replan
            replan_envs = (
                torch.logical_and(
                    (self.decimation_counter == self.planner_decimation),
                    self._obs_env_step_counter >= self.model.cfg.prediction_horizon,
                )
                .nonzero()
                .squeeze(1)
                .tolist()
            )
            # get the environments where the buffer is not filled yet
            replan_not_filled = (
                torch.logical_and(
                    (self.decimation_counter == self.planner_decimation),
                    self._obs_env_step_counter < self.model.cfg.prediction_horizon,
                )
                .nonzero()
                .squeeze(1)
                .tolist()
            )

            # resample when the buffer filled for the first time
            obs["planner_obs"]["resample_population"] = self._obs_env_step_counter == self.model.cfg.prediction_horizon

            # se2_positions_w is return with shape (num_envs, 1, trajectory_length, state_dim) where state_dim is 3
            # se2_velocity_b is return with shape (num_envs, trajectory_length, action_dim) where action_dim is 3
            # obs["planner_obs"]["resample_population"] = env_reset

            # also resample the population for standing environments
            non_moving_samples = self._non_moving_step_counter > self.cfg.movement_resample_count
            obs["planner_obs"]["resample_population"] |= non_moving_samples
            if torch.any(non_moving_samples):
                print(f"[INFO]: Resampling population of env {torch.where(non_moving_samples)[0]} bc non-moving state.")

            # planning
            if len(replan_envs) > 0:
                if torch.any(obs["planner_obs"]["resample_population"]):
                    print(
                        "[INFO]: Resampling population for"
                        f" {torch.where(obs['planner_obs']['resample_population'])[0].tolist()}"
                    )
                    print(
                        "[INFO]: Buffer step idx"
                        f" {self._obs_env_step_counter[obs['planner_obs']['resample_population'].cpu()].tolist()}"
                    )

                # add states, proprio and extero observations to the planner obs
                obs["planner_obs"]["states"] = self._state_history.clone()
                obs["planner_obs"]["proprio_obs"] = self._proprio_obs_history.clone()
                obs["planner_obs"]["extero_obs"] = obs["fdm_obs_exteroceptive"].clone()

                se2_positions_w_pred, se2_velocity_b[replan_envs] = self.planner.plan(
                    obs=obs["planner_obs"], env_ids=replan_envs
                )
                se2_positions_w[replan_envs] = se2_positions_w_pred[:, 0, :, :]

            if len(replan_not_filled) > 0:
                # print(f"[INFO]: Resampling population for not filled buffer {replan_not_filled}")
                # print(f"[INFO]: Buffer step idx {self._obs_env_step_counter[replan_not_filled].tolist()}")

                # replace actions if obs buffer has not been filled yet
                se2_velocity_b[replan_not_filled] = torch.roll(initial_rand_actions[replan_not_filled], 1, dims=1)
                se2_positions_w[replan_not_filled] *= 0

            # update decimation counter
            self.decimation_counter[replan_envs] = 0
            self.decimation_counter[replan_not_filled] = 0
            self.decimation_counter += 1
            # check if applied command is lower than threshold
            self._non_moving_step_counter += torch.norm(se2_velocity_b[:, 0, :], dim=-1) < self.cfg.movement_threshold
            self._non_moving_step_counter[
                torch.norm(se2_velocity_b[:, 0, :], dim=-1) >= self.cfg.movement_threshold
            ] = 0

            ###
            # Visualize the selected path
            ###
            if self.draw_interface is not None:
                self._visualize(replan_envs, se2_positions_w, se2_velocity_b, obs)

            ###
            # Capture image if cameras are given
            ###

            if cameras is not None:
                for idx, camera in enumerate(cameras):
                    if goal_ever_reached[idx] or self._obs_env_step_counter[idx] < self.model.cfg.prediction_horizon:
                        continue
                    camera.get_current_frame()
                    # Convert RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(camera.get_rgba()[:, :, :3], cv2.COLOR_RGB2BGR)
                    # Save the image as PNG
                    assert cv2.imwrite(
                        f"{cam_save_path[idx]}/img_{str(render_counter[idx].item()).zfill(4)}.png", image_bgr
                    )

                    render_counter[idx] += 1

                # update goal ever reached, if yes, then stop recording
                goal_ever_reached |= goal_reached

                # break if all environments are done
                if torch.all(goal_ever_reached) or torch.any(render_counter >= 1000):
                    break

        if cameras is not None:
            print(f"[INFO]: Images saved to {cam_save_path}. Generating video.")
            for idx, path in enumerate(cam_save_path):
                os.system(
                    f"ffmpeg -r {int(1 / self.env.step_dt)} -f image2 -s 1920x1080 -i"
                    f" '{path}/img_%04d.png' -vcodec libx264 -profile:v high -crf 25 -pix_fmt yuv420p"
                    f" '{path}/video.mp4'"
                )

    def close(self):
        del self.planner
        del self.model
        self.env.close()
        del self.env

    """
    Observation Buffers
    """

    def _init_obs_buffers(self):
        # local history buffers
        assert isinstance(self.env.observation_manager.group_obs_dim["fdm_state"], tuple)
        self._state_history = torch.zeros(
            (
                self.env.num_envs,
                self.model.cfg.history_length,
                *(self.env.observation_manager.group_obs_dim["fdm_state"]),
            ),
            device=self.device,
        )
        assert isinstance(self.env.observation_manager.group_obs_dim["fdm_obs_proprioception"], tuple)
        self._proprio_obs_history = torch.zeros(
            (
                self.env.num_envs,
                self.model.cfg.history_length,
                *(self.env.observation_manager.group_obs_dim["fdm_obs_proprioception"]),
            ),
            device=self.device,
        )

        # init buffer for obs_env_step_counter
        self._obs_env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int)

        # get the interval to record the local history
        self._history_collection_interval = (
            self.model.cfg.command_timestep / self.env.step_dt / self.model.cfg.history_length
        )

        # init buffer to store past positions for visualization
        self._past_positions = torch.zeros(
            self.env.num_envs,
            int(self.cfg.max_path_time / (self._history_collection_interval * self.env.step_dt)),
            3,
            device=self.device,
        )
        self._past_positions_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int)

    def _update_obs_buffers(
        self, state: torch.Tensor, obersevations_proprioceptive: torch.Tensor, feet_contact: torch.Tensor
    ):
        # local robot state history buffer
        updatable_envs = (self._obs_env_step_counter % self._history_collection_interval).type(  # noqa: E721
            torch.int
        ) == 0
        # don't update if robot has not touched the ground yet (initial falling period after reset)
        updatable_envs[~feet_contact] = False

        # write the current robot state into the buffer
        self._state_history[updatable_envs] = torch.roll(self._state_history[updatable_envs], 1, dims=1)
        self._state_history[updatable_envs, 0] = state[updatable_envs]

        # write the current proprioceptive observation into the buffer
        self._proprio_obs_history[updatable_envs] = torch.roll(self._proprio_obs_history[updatable_envs], 1, dims=1)
        self._proprio_obs_history[updatable_envs, 0] = obersevations_proprioceptive[updatable_envs]

        # past positions of the robot
        self._past_positions[updatable_envs, self._past_positions_counter[updatable_envs]] = state[updatable_envs, :3]
        self._past_positions_counter[updatable_envs] += 1

        # update step counter for all environments
        # NOTE: we only start the counting once all feet were in contact
        self._obs_env_step_counter[feet_contact] += 1

    def _reset_obs_buffers(self, env_ids: torch.Tensor):
        # reset the step counter
        self._obs_env_step_counter[env_ids] *= 0
        # reset the past positions counter
        self._past_positions_counter[env_ids] *= 0

        # reset the buffer for the given environments
        self._state_history[env_ids] *= 0
        self._proprio_obs_history[env_ids] *= 0
        self._past_positions[env_ids] *= 0

    """
    Helper functions
    """

    def _action_transformer(self, se2_positions_w: torch.Tensor, se2_velocity_b: torch.Tensor) -> torch.Tensor:
        # reshape to (num_envs x trajectory_length, 3)
        TRAJ_DIM = se2_velocity_b.shape[1]
        se2_velocity_b = se2_velocity_b.reshape(-1, se2_velocity_b.shape[-1])
        se2_quat_w = se2_positions_w.reshape(-1, se2_positions_w.shape[-1])[:, 2]
        # extract the base frame rotation as each point of the trajectory
        base_quat_w = math_utils.quat_from_euler_xyz(
            roll=torch.zeros_like(se2_quat_w), pitch=torch.zeros_like(se2_quat_w), yaw=se2_quat_w
        )
        # transform the velocity commands into world frame
        lin_vel_b = torch.zeros((se2_velocity_b.shape[0], 3), device=self.env.device)
        lin_vel_b[:, :2] = se2_velocity_b[:, :2]
        lin_vel_w = math_utils.quat_rotate_inverse(base_quat_w, lin_vel_b)
        ang_vel_b = torch.zeros((se2_velocity_b.shape[0], 3), device=self.env.device)
        ang_vel_b[:, 2] = se2_velocity_b[:, 2]
        ang_vel_w = math_utils.quat_rotate_inverse(base_quat_w, ang_vel_b)
        return torch.hstack((lin_vel_w[:, :2], ang_vel_w[:, 2].unsqueeze(1))).reshape(-1, TRAJ_DIM, 3)

    def _planner_eval_metrics(
        self,
        finished_path_counter: int,
        path_rrt_length: torch.Tensor,
        traversed_path_length: torch.Tensor,
        goal_success: torch.Tensor,
        trajectory_time: torch.Tensor,
    ) -> dict:
        # success rate, fail rate and number of paths
        metrics = {
            "Finished Paths": {
                "Success": goal_success[:finished_path_counter].float().mean().item(),
                "Fail": (~goal_success[:finished_path_counter]).float().mean().item(),
                "All": finished_path_counter,
            }
        }
        # SPL
        spl = (
            goal_success[:finished_path_counter].float()
            * path_rrt_length[:finished_path_counter]
            / torch.max(path_rrt_length[:finished_path_counter], traversed_path_length[:finished_path_counter])
        )
        metrics["SPL"] = {
            "Success": spl[goal_success[:finished_path_counter]].mean().item(),
            "Fail": spl[~goal_success[:finished_path_counter]].mean().item(),
            "All": spl.mean().item(),
        }
        # mean trajectory time
        mean_trajectory_time = trajectory_time[:finished_path_counter] * self.env.step_dt
        metrics["Mean Trajectory Time"] = {
            "Success": mean_trajectory_time[goal_success[:finished_path_counter]].mean().item(),
            "Fail": mean_trajectory_time[~goal_success[:finished_path_counter]].mean().item(),
            "All": mean_trajectory_time.mean().item(),
        }
        # mean trajectory length
        mean_trajectory_length = traversed_path_length[:finished_path_counter]
        metrics["Mean Trajectory Length"] = {
            "Success": mean_trajectory_length[goal_success[:finished_path_counter]].mean().item(),
            "Fail": mean_trajectory_length[~goal_success[:finished_path_counter]].mean().item(),
            "All": mean_trajectory_length.mean().item(),
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: dict[str, dict[str, dict[str, float]]]):
        """Print the metrics of the planner evaluation."""
        # print final evaluation
        table = prettytable.PrettyTable()
        table.field_names = ["Metric", "Success", "Fail", "All"]
        for key, value in metrics.items():
            table.add_row([key, value["Success"], value["Fail"], value["All"]])
        print("[INFO] Final Planner Evaluation\n", table)

    """
    Visualization
    """

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.command_vel_visualizer[idx].cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        return arrow_scale, arrow_quat

    def _visualize(self, replan_envs: list[int], pred_path: torch.Tensor, actions: torch.Tensor, obs: dict):
        """Visualize the planner."""
        self.draw_interface.clear_lines()
        # enable type hints
        window_interface: PlannerEnvWindow = self.env._window
        if window_interface.current_cost_viz_mode == "Cost":
            if len(replan_envs) > 0:
                self.population_states[replan_envs] = self.planner.states.clone()
                self.population_costs[replan_envs] = self.planner.total_cost.clone()

            # visualize the paths
            self._visualize_path_cost(self.population_states, self.cost_population_colors, self.population_costs)

        elif window_interface.current_cost_viz_mode == "Pose Reward":
            if len(replan_envs) > 0:
                self.population_goal_reward[replan_envs] = self.planner.pose_cost.clone()
                self.population_states[replan_envs] = self.planner.states.clone()

            # visualize the paths
            self._visualize_path_cost(self.population_states, self.cost_population_colors, self.population_goal_reward)

        elif window_interface.current_cost_viz_mode == "Goal Distance":
            if len(replan_envs) > 0:
                self.population_goal_offset[replan_envs] = self.planner.position_offset.clone()
                self.population_states[replan_envs] = self.planner.states.clone()

            # visualize the paths
            self._visualize_path_cost(self.population_states, self.cost_population_colors, self.population_goal_offset)

        elif window_interface.current_cost_viz_mode == "Goal Distance X":
            if len(replan_envs) > 0:
                self.population_states[replan_envs] = self.planner.states.clone()

            # calc the x distance to the goal
            self.population_goal_offset = torch.abs(
                self.population_states[:, :, -1, 0] - obs["planner_obs"]["goal"][:, None, 0]
            )

            # visualize the paths
            self._visualize_path_cost(self.population_states, self.cost_population_colors, self.population_goal_offset)

        elif window_interface.current_cost_viz_mode == "Goal Distance Y":
            if len(replan_envs) > 0:
                self.population_states[replan_envs] = self.planner.states.clone()

            # calc the y distance to the goal
            self.population_goal_offset = torch.abs(
                self.population_states[:, :, -1, 1] - obs["planner_obs"]["goal"][:, None, 1]
            )

            # visualize the paths
            self._visualize_path_cost(self.population_states, self.cost_population_colors, self.population_goal_offset)

        elif window_interface.current_cost_viz_mode == "Collision":
            if len(replan_envs) > 0:
                self.population_states[replan_envs] = self.planner.states.clone()
                self.population_collision_costs[replan_envs] = self.planner.collision_traj_cost.clone()

            # visualize the paths
            self._visualize_path_cost(
                self.population_states,
                self.coll_population_colors,
                self.population_collision_costs,
                binary_color_cost_threshold=self.planner.to_cfg.collision_cost_high_risk_factor,
            )

        elif window_interface.current_cost_viz_mode == "Height Scan Cost":
            if len(replan_envs) > 0:
                self.population_states[replan_envs] = self.planner.states.clone()
                self.population_costs[replan_envs] = self.planner.curr_cost_map_cost.clone()

            # visualize the paths
            self._visualize_path_cost(
                self.population_states,
                self.coll_population_colors,
                self.population_costs,
                binary_color_cost_threshold=self.planner.to_cfg.state_cost_w_fatal_trav,
            )

        # perfect velocity visualization estimation
        if window_interface.perfect_velocity:
            if len(replan_envs) > 0:
                self.population_perfect_velocity[replan_envs] = self.planner.b_obj_func_N_step(
                    self.planner.population, only_rollout=True, control_mode="velocity_control"
                )

            # visualize the paths
            self._visualize_path_cost(self.population_perfect_velocity, self.cost_population_colors, offset=-0.15)

        # smooth paths
        smoothed_path = pp.chspline(
            pred_path[:, self.cfg.spline_smooth_n :: self.cfg.spline_smooth_n, :],
            interval=1.0 / self.cfg.spline_smooth_n,
        )

        for env_idx, path in enumerate(smoothed_path):
            path[..., 2] = self.env.scene.articulations["robot"].data.root_pos_w[env_idx, 2].item() + 0.1
            self.draw_interface.draw_lines(
                path[:-1].tolist(),
                path[1:].tolist(),
                [(0, 0, 1, 1)] * int(path.shape[0] - 1),
                [10] * int(path.shape[0] - 1),
            )
        for env_idx, path in enumerate(pred_path):
            path[..., 2] = self.env.scene.articulations["robot"].data.root_pos_w[env_idx, 2].item() + 0.1
            self.draw_interface.draw_lines(
                path[:-1].tolist(),
                path[1:].tolist(),
                [(0, 0, 0.5, 1)] * int(path.shape[0] - 1),
                [10] * int(path.shape[0] - 1),
            )

        # transform veloicty commands into base frame and declare them as actions
        se2_velocity_w = self._action_transformer(se2_positions_w=pred_path, se2_velocity_b=actions)

        # set markers
        for traj_idx, marker in enumerate(self.command_vel_visualizer):
            # -- resolve the scales and quaternions
            vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
                se2_velocity_w[:, traj_idx, :2], traj_idx
            )
            # set the z height
            se2_pos_with_z = pred_path[:, traj_idx].clone()
            se2_pos_with_z[:, 2] = self.env.scene.articulations["robot"].data.root_pos_w[:, 2] + 0.1
            # display markers
            marker.visualize(se2_pos_with_z, vel_arrow_quat, vel_arrow_scale)

        # plot trajectories from past position buffer
        for env_idx in range(self.env.num_envs):
            if self._past_positions_counter[env_idx] <= 1:
                continue
            self.draw_interface.draw_lines(
                self._past_positions[env_idx, : self._past_positions_counter[env_idx]].tolist()[:-1],
                self._past_positions[env_idx, : self._past_positions_counter[env_idx]].tolist()[1:],
                [(0.63, 0.13, 0.95, 1)] * (self._past_positions_counter[env_idx] - 1),
                [5.0] * (self._past_positions_counter[env_idx] - 1),
            )

    def _visualize_path_cost(
        self,
        states: torch.Tensor,
        colors: list[tuple[float, float, float, float]],
        cost: torch.Tensor | None = None,
        binary_color_cost_threshold: float | None = None,
        offset: float = 0.0,
    ):
        # get number of paths per color
        traj_per_color = math.ceil(self.planner_cfg["optim"]["population_size"] / len(colors))

        # get the pairs of the states to draw the lines
        states_start = states[:, :, :-1, :].reshape(-1, 3).cpu()
        states_end = states[:, :, 1:, :].reshape(-1, 3).cpu()

        # extract terrain analysis module from goal command generator
        analysis: TerrainAnalysis = self.env.command_manager._terms["command"].analysis

        # get the height of the terrain at the start and end position
        states_start[:, 2] = analysis.get_height(states_start[:, :2])
        states_end[:, 2] = analysis.get_height(states_end[:, :2])

        # add root robot height and offset
        states_start[:, 2] += self.env.scene.articulations["robot"].data.root_pos_w[0, 2].item() + offset
        states_end[:, 2] += self.env.scene.articulations["robot"].data.root_pos_w[0, 2].item() + offset

        # sort the cost and assign the colors if cost is not None
        if cost is None:
            color_idx = torch.zeros(states_start.shape[0], dtype=torch.int)
        elif binary_color_cost_threshold is not None:
            colors_tensor = torch.tensor(colors)
            color_idx = torch.zeros(cost.shape[0], cost.shape[1], dtype=torch.int, device=cost.device)
            color_idx[cost >= binary_color_cost_threshold] = 1
            # expand color_idx to traj_len and flatten
            color_idx = color_idx.unsqueeze(-1).repeat(1, 1, states.shape[2] - 1).view(-1)
        else:
            cost_sort = torch.argsort(cost, dim=1)
            color_idx = torch.zeros(cost.shape[0], cost.shape[1], dtype=torch.int, device=cost_sort.device)
            env_ids = torch.arange(cost.shape[0], device=cost_sort.device)
            env_ids_traj_per_color = env_ids.clone()[:, None].repeat(1, traj_per_color)
            for i in range(len(colors)):
                curr_cost_sort = cost_sort[env_ids, traj_per_color * i : traj_per_color * (i + 1)]
                color_idx[env_ids_traj_per_color[:, : curr_cost_sort.shape[1]], curr_cost_sort] = i
            colors_tensor = torch.tensor(colors)
            if self.planner.to_cfg.control == "fdm":
                # all red color if cost higher than threshold
                color_idx[cost > self.planner.to_cfg.collision_cost_high_risk_factor] = len(colors) - 1
            # expand color_idx to traj_len and flatten
            color_idx = color_idx.unsqueeze(-1).repeat(1, 1, states.shape[2] - 1).view(-1)

        color = colors_tensor[color_idx.cpu()]

        # draw the lines
        self.draw_interface.draw_lines(
            states_start.tolist(),
            states_end.tolist(),
            color.tolist(),
            [5.0] * states_end.shape[0],
        )
