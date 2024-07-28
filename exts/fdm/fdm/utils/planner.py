

from __future__ import annotations

import os
import prettytable
import statistics
import torch

import hydra
import omegaconf
import pypose as pp

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils.io import dump_yaml

from fdm.mdp import GoalCommand
from fdm.sampling_planner import SimpleSE2TrajectoryOptimizer
from omni.isaac.lab_tasks.utils import get_checkpoint_path

from ..model import FDMModel
from .planner_cfg import FDMPlannerCfg
from .replay_buffer import ReplayBuffer
from .trajectory_dataset import TrajectoryDataset
from .utils import generate_colors

# can only be imported if gui activated
try:
    from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
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

        # separate hue value for the number of runs (safe + collision and path color)
        colors = [
            generate_colors(
                self.nb_draw_traj,
                start_hue=0.33 * step - 0.25 * 0.33,
                end_hue=0.33 * step + 0.25 * 0.33,
            )
            for step in range(1, 4)
        ]
        self.safe_colors = colors[0]
        self.collision_colors = colors[1]
        self.trajectory_color = colors[2]

        # init debug draw
        if omni_debug_draw:
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        else:
            raise RuntimeError("Debug draw interface not available. Run in `gui` mode.")
        # init debug markers for selected velocity commands
        self.command_vel_visualizer: list[VisualizationMarkers] = []
        for traj_idx in range(self.planner_cfg["traj_dim"]):
            marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/velocity_command"
            marker_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)
            self.command_vel_visualizer.append(VisualizationMarkers(marker_cfg))
            self.command_vel_visualizer[traj_idx].set_visibility(True)

        # override the resampling command of the command generator with `trainer_cfg.command_timestep`
        self.cfg.env_cfg.episode_length_s = self.cfg.model_cfg.command_timestep * (
            self.cfg.replay_buffer_cfg.trajectory_length + 1
        )

        # setup
        self.setup()

        # resume model and set into eval mode
        self.log_root_path = os.path.join("logs", "fdm", self.cfg.experiment_name)
        self.log_root_path = os.path.abspath(self.log_root_path)
        resume_path = get_checkpoint_path(self.log_root_path, self.cfg.load_run, self.cfg.load_checkpoint)
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
        self.model: FDMModel = FDMModel(cfg=self.cfg.model_cfg, device=self.device)
        self.model.to(self.device)
        # setup trainer
        cfg = omegaconf.OmegaConf.create(self.planner_cfg)
        self.planner: SimpleSE2TrajectoryOptimizer = hydra.utils.instantiate(cfg.to)
        # setup replay buffer
        self.replay_buffer = ReplayBuffer(self.cfg.replay_buffer_cfg, self.model.cfg, self.env)

        print("[INFO]: Setup complete.")

    def navigate(self):
        """Run the visual evaluation of the model.

        Args:
            initial_warm_up (bool, optional): Let the environments run until the history buffers are filled. Then
                performs a first prediciton and return the results. Defaults to True.

        """
        # reset the environment
        with torch.inference_mode():
            obs, _ = self.env.reset(1)
        # get initial actions
        # se2_positions_w is return with shape (num_envs, 1, trajectory_length, state_dim) where state_dim is 3
        # se2_velocity_w is return with shape (num_envs, trajectory_length, action_dim) where action_dim is 3
        obs["planner_obs"]["resample_population"] = True
        se2_positions_w, se2_velocity_b = self.planner.plan(obs=obs["planner_obs"])
        se2_velocity_w = self._action_transformer(se2_positions_w=se2_positions_w, se2_velocity_b=se2_velocity_b)

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

        # buffer to save trajectories for plotting
        pred_trajectories = {x: [] for x in range(self.env.num_envs)}
        pred_collision = {x: [] for x in range(self.env.num_envs)}
        final_pred_error = {x: [] for x in range(self.env.num_envs)}
        # step counter
        counter = 0
        meta_eval: dict[str, list[float]] = {}

        # get the feet index in the contact sensor
        feet_idx, _ = self.env.scene.sensors["contact_forces"].find_bodies(".*FOOT")
        feet_contact = torch.zeros((self.env.num_envs, len(feet_idx)), dtype=torch.bool, device=self.env.device)

        while ~goal_command.all_path_completed:
            # step environment
            with torch.inference_mode():
                # NOTE: goal reached is the only source for an time-out
                obs, _, dones, goal_reached, _ = self.env.step(se2_velocity_b[:, 0, :])

            ###
            # Get the actions
            ###
            # se2_positions_w is return with shape (num_envs, 1, trajectory_length, state_dim) where state_dim is 3
            # se2_velocity_b is return with shape (num_envs, trajectory_length, action_dim) where action_dim is 3
            obs["planner_obs"]["resample_population"] = False
            se2_positions_w, se2_velocity_b = self.planner.plan(obs=obs["planner_obs"])
            se2_positions_w = se2_positions_w[:, 0, :, :]

            # smooth actions
            # se2_velocity_b = pp.chspline(
            #     se2_velocity_b[:, :: self.cfg.spline_smooth_n, :], interval=1.0 / self.cfg.spline_smooth_n
            # )

            # smooth paths
            smoothed_path = pp.chspline(
                se2_positions_w[:, self.cfg.spline_smooth_n :: self.cfg.spline_smooth_n, :],
                interval=1.0 / self.cfg.spline_smooth_n,
            )

            # transform veloicty commands into base frame and declare them as actions
            se2_velocity_w = self._action_transformer(se2_positions_w=se2_positions_w, se2_velocity_b=se2_velocity_b)

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
            self.draw_interface.clear_lines()
            for env_idx, path in enumerate(smoothed_path):
                path[..., 2] = self.env.scene.articulations["robot"].data.root_pos_w[env_idx, 2].item()
                self.draw_interface.draw_lines(
                    path[:-1].tolist(),
                    path[1:].tolist(),
                    [(1, 0, 0, 1)] * int(path.shape[0] - 1),
                    [3] * int(path.shape[0] - 1),
                )
            for env_idx, path in enumerate(se2_positions_w):
                path[..., 2] = self.env.scene.articulations["robot"].data.root_pos_w[env_idx, 2].item()
                self.draw_interface.draw_lines(
                    path[:-1].tolist(),
                    path[1:].tolist(),
                    [(0, 1, 0, 1)] * int(path.shape[0] - 1),
                    [3] * int(path.shape[0] - 1),
                )

            # set markers
            for traj_idx, marker in enumerate(self.command_vel_visualizer):
                # -- resolve the scales and quaternions
                vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
                    se2_velocity_w[:, traj_idx, :2], traj_idx
                )
                # display markers
                marker.visualize(se2_positions_w[:, traj_idx], vel_arrow_quat, vel_arrow_scale)

            ###
            # Print eval info
            ###

            debug_path_counter += done_updated.sum() + goal_reached_updated.sum()
            # for every 100 paths, given current stats
            if debug_path_counter > 100:
                debug_path_counter = 0
                table = prettytable.PrettyTable()
                table.field_names = ["Metric", "Success", "Fail", "All"]
                for key, value in self._planner_eval_metrics(
                    finished_path_counter, path_rrt_length, traversed_path_length, goal_success, trajectory_time
                ).items():
                    table.add_row([key, value["Success"], value["Fail"], value["All"]])
                print("[INFO] Planner Evaluation\n", table)

            ###########################################################################################################
            # FDM Evaluation
            ###########################################################################################################

            # FIXME: currently skip this part until planne works
            continue

            if self.replay_buffer.is_filled:
                continue

            # also mark every env as done where the replay buffer is filled and where the goal is reached
            # TODO: technically does not collide when the goal is reached, special case, should be handled differently
            dones = dones | self.replay_buffer.env_buffer_filled.to(self.device) | goal_reached

            ###
            # Determine feet contact
            ###
            # Note: only start recording and changing actions when all feet have touched the ground
            feet_contact[
                torch.norm(self.env.scene.sensors["contact_forces"].data.net_forces_w[:, feet_idx], dim=-1) > 1
            ] = True
            feet_all_contact = torch.all(feet_contact, dim=-1)
            # reset for terminated envs
            feet_all_contact[dones] = False

            ###
            # update replay buffer and get completed predictions
            ###
            dones = dones.to(self.replay_buffer.device)
            self.replay_buffer.add(
                states=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                obersevations_exteroceptive=obs["fdm_obs_exteroceptive"].clone(),
                actions=se2_velocity_b.clone(),
                dones=dones.to(torch.bool).clone(),
                feet_contact=feet_all_contact,
            )
            if torch.any(dones):
                # for done environments reset replay_buffer
                self.replay_buffer.reset(env_ids=self.replay_buffer._ALL_INDICES[dones])
                # reset saved trajectories
                [pred_trajectories[env_id.item()].clear() for env_id in self.replay_buffer._ALL_INDICES[dones]]
                [pred_collision[env_id.item()].clear() for env_id in self.replay_buffer._ALL_INDICES[dones]]
                [final_pred_error[env_id.item()].clear() for env_id in self.replay_buffer._ALL_INDICES[dones]]
            # decide for which environments to do a new prediction
            # note: first time prediction can be done when fill idx is increased to 1, next when it reaches 1+prediction_horizon, etc.
            env_new_prediction = self.replay_buffer._ALL_INDICES[
                self.replay_buffer.env_step_counter % int(self.replay_buffer.data_collection_interval) == 0
            ]
            env_new_prediction = env_new_prediction[self.replay_buffer.fill_idx[env_new_prediction] > 0]
            if not torch.any(env_new_prediction):
                continue

            ###
            # Loss, Drawing and Info print
            ###
            # clear previous drawings
            self.draw_interface.clear_lines()

            # update predictions and save them for plotting
            future_states, collision_pred = self._eval_predict(env_new_prediction)
            # calculate loss
            pred_trajectories, pred_collision, final_pred_error, meta_eval, counter = self._eval_loss(
                env_new_prediction,
                future_states,
                collision_pred,
                pred_trajectories,
                pred_collision,
                final_pred_error,
                meta_eval,
                counter,
            )
            # update drawing of prediction and walked trajectories
            self._draw_trajectories(
                pred_trajectories,
                pred_collision,
                final_pred_error,
                safe_colors=self.safe_colors,
                collision_colors=self.collision_colors,
                draw_trajectory=True,
            )
            # print loss information
            if counter > 1000:
                # print losses as table
                table = prettytable.PrettyTable()
                table.field_names = [f"Loss ({counter} predictions)", "Mean Value", "Std Value"]
                for key, value in meta_eval.items():
                    table.add_row([key, statistics.mean(value[5:]), statistics.stdev(value[5:])])
                print(table)
                # reset loss
                meta_eval = {}
                # reset counter
                counter = 0

        # get final eval metrics
        metrics = self._planner_eval_metrics(
            finished_path_counter, path_rrt_length, traversed_path_length, goal_success, trajectory_time
        )

        # print final evaluation
        table = prettytable.PrettyTable()
        table.field_names = ["Metric", "Success", "Fail", "All"]
        for key, value in metrics.items():
            table.add_row([key, value["Success"], value["Fail"], value["All"]])
        print("[INFO] Final Planner Evaluation\n", table)

        # dump results into yaml
        dump_yaml(self.log_root_path + "/eval_metric_sampling_planner.yaml", metrics)

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

    def _eval_predict(self, env_ids: torch.Tensor, model: torch.nn.Module | None = None):
        """Make predictions based on the current states and the planned actions"""

        # get initial states
        initial_states = self.replay_buffer.states[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, 0][:, None, :7]
        initial_states_SE3 = pp.SE3(
            initial_states.repeat(1, self.cfg.model_cfg.prediction_horizon, 1).reshape(-1, 7)
        ).to(self.device)

        # get state history transformed into local frame
        state_history = TrajectoryDataset.state_history_transformer(
            self.replay_buffer,
            torch.vstack([env_ids, self.replay_buffer.fill_idx[env_ids] - 1]).T,
            initial_states,
            self.model.cfg.history_length,
        ).to(self.device)
        # collect future actions
        future_actions = torch.concatenate(
            [
                self.agent._plan[env_ids, self.agent._plan_step[env_ids] - 1 + idx][:, None, :]
                for idx in range(self.cfg.model_cfg.prediction_horizon)
            ],
            dim=1,
        ).to(self.device)

        # make predictions
        model_in = (
            state_history,
            self.replay_buffer.observations_proprioceptive[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, :].to(
                self.device
            ),
            self.replay_buffer.observations_exteroceptive[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, :]
            .type(torch.float32)
            .to(self.device),
            future_actions,
        )
        if model:
            with torch.no_grad():
                future_states, collision_prob = model.forward(model_in)
        else:
            future_states, collision_prob = self.trainer.predict(model_in)

        # transform future states in global frame
        future_states = torch.concatenate(
            [
                future_states[..., :2],
                torch.zeros_like(future_states[..., 2]).unsqueeze(2),
                math_utils.convert_quat(
                    math_utils.quat_from_euler_xyz(
                        roll=torch.zeros_like(future_states[..., 2]),
                        pitch=torch.zeros_like(future_states[..., 2]),
                        yaw=torch.atan2(future_states[..., 2], future_states[..., 3]),
                    ),
                    to="xyzw",
                ),
            ],
            dim=-1,
        )
        future_states = (initial_states_SE3 * pp.SE3(future_states.reshape(-1, 7))).tensor()
        future_states_yaw = math_utils.euler_xyz_from_quat(future_states[..., [6, 3, 4, 5]])[2]
        future_states = torch.concatenate([future_states[..., :3], future_states_yaw.unsqueeze(1)], dim=-1).reshape(
            -1, self.cfg.model_cfg.prediction_horizon, 4
        )

        return future_states, collision_prob

    def _eval_loss(
        self,
        env_new_prediction: torch.Tensor,
        future_states: torch.Tensor,
        collision_pred: torch.Tensor,
        pred_trajectories: dict[int, list[torch.Tensor]],
        pred_collision: dict[int, list[torch.Tensor]],
        final_pred_error: dict[int, list[torch.Tensor]],
        meta_eval: dict[str, list[float]],
        counter: int,
    ) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]], dict[str, list[float]], int]:
        # loss for world frame coordinates
        for env_idx in env_new_prediction:
            # get previous predictions
            prev_pred = pred_trajectories[env_idx.item()]
            prev_coll = pred_collision[env_idx.item()]
            if len(prev_pred) <= self.cfg.model_cfg.prediction_horizon:
                continue
            # NOTE: the predicitions also include the initial position for the transformation, has to be removed here
            prev_pred = prev_pred[-self.cfg.model_cfg.prediction_horizon][-self.cfg.model_cfg.prediction_horizon :]
            if self.cfg.model_cfg.unified_failure_prediction:
                prev_coll = prev_coll[-self.cfg.model_cfg.prediction_horizon]
            else:
                prev_coll = prev_coll[-self.cfg.model_cfg.prediction_horizon][-self.cfg.model_cfg.prediction_horizon :]

            # get loss values
            model_out = [prev_pred[:, [0, 1, 2, 3]].unsqueeze(0), prev_coll.unsqueeze(0)]
            future_states_yaw = math_utils.euler_xyz_from_quat(
                self.replay_buffer.states[
                    env_idx,
                    self.replay_buffer.fill_idx[env_idx]
                    - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                    0,
                    [6, 3, 4, 5],
                ].reshape(-1, 4)
            )[2].reshape(-1, 1)
            # encode yaw with sin and cos
            future_states_yaw = torch.cat((torch.sin(future_states_yaw), torch.cos(future_states_yaw)), dim=1)
            target = (
                torch.hstack((
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        :2,
                    ],
                    future_states_yaw,
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        -1,
                    ].unsqueeze(1),
                ))
                .unsqueeze(0)
                .to(self.device)
            )

            # compute loss
            _, meta = self.model.loss(model_out, target, mode="eval")
            if len(meta_eval) == 0:
                for key, value in meta.items():
                    meta_eval[key] = [value]
            else:
                for key, value in meta.items():
                    meta_eval[key].append(value)

            # save final prediction error
            final_pred_error[env_idx.item()].append(
                torch.vstack((prev_pred[-1, :3], torch.cat((target[0, -1, :2], prev_pred[-1, 2].unsqueeze(0)))))
            )

            # increase counter
            counter += target.shape[0]

        # append initial position to future states
        future_states = torch.cat(
            (
                self.replay_buffer.states[env_new_prediction, self.replay_buffer.fill_idx[env_new_prediction] - 1, 0][
                    :, None, [0, 1, 2, 7]
                ].to(self.device),
                future_states,
            ),
            dim=1,
        )
        [
            pred_trajectories[env_id.item()].append(future_states[pred_idx].clone())
            for pred_idx, env_id in enumerate(env_new_prediction)
        ]
        [
            pred_collision[env_id.item()].append(collision_pred[pred_idx].clone())
            for pred_idx, env_id in enumerate(env_new_prediction)
        ]
        return pred_trajectories, pred_collision, final_pred_error, meta_eval, counter

    def _draw_trajectories(
        self,
        pred_trajectories: dict[int, list[torch.Tensor]],
        pred_collision: dict[int, list[torch.Tensor]],
        final_pred_error: dict[int, list[torch.Tensor]],
        safe_colors: list[tuple[float]],
        collision_colors: list[tuple[float]],
        draw_trajectory: bool = True,
    ):
        if not hasattr(self, "draw_interface"):
            raise RuntimeError("Draw interface not initialized. Set `eval=True` in the runner constructor.")

        for env_idx in range(self.env.num_envs):
            if draw_trajectory:
                start_idx = (
                    0 if self.replay_buffer.fill_idx[env_idx] < 10 else self.replay_buffer.fill_idx[env_idx] - 10
                )
                # plot trajectories from replay buffer
                self.draw_interface.draw_lines(
                    self.replay_buffer.states[
                        env_idx, start_idx : self.replay_buffer.fill_idx[env_idx], 0, :3
                    ].tolist()[:-1],
                    self.replay_buffer.states[
                        env_idx, start_idx : self.replay_buffer.fill_idx[env_idx], 0, :3
                    ].tolist()[1:],
                    self.trajectory_color * (self.replay_buffer.fill_idx[env_idx] - 1 - start_idx),
                    [5.0] * (self.replay_buffer.fill_idx[env_idx] - 1 - start_idx),
                )

            # plot every prediction made for the environment
            if len(pred_trajectories[env_idx]) > self.nb_draw_traj * self.step_draw_traj:
                pred_iter = pred_trajectories[env_idx][-self.nb_draw_traj * self.step_draw_traj :]
                pred_iter = pred_iter[::2] if len(pred_trajectories[env_idx]) % 2 == 0 else pred_iter[1::2]
                coll_iter = pred_collision[env_idx][-self.nb_draw_traj * self.step_draw_traj :]
                coll_iter = coll_iter[::2] if len(pred_trajectories[env_idx]) % 2 == 0 else coll_iter[1::2]
                error_iter = final_pred_error[env_idx][-self.nb_draw_traj * (self.step_draw_traj - 1) :]
                error_iter = error_iter[::2] if len(pred_trajectories[env_idx]) % 2 == 0 else error_iter[1::2]
            else:
                pred_iter = pred_trajectories[env_idx][::2]
                coll_iter = pred_collision[env_idx][::2]
                error_iter = final_pred_error[env_idx][::2]

            for pred_idx, pred in enumerate(pred_iter):
                # in collision color is red otherwise green
                color = collision_colors if torch.any(coll_iter[pred_idx] > 0.5) else safe_colors
                self.draw_interface.draw_lines(
                    pred[:-1, :3].tolist(),
                    pred[1:, :3].tolist(),
                    color,
                    [5.0] * (pred.shape[0] - 1),
                )

            if len(error_iter) > 0:
                for curr_error in error_iter:
                    self.draw_interface.draw_lines(
                        [curr_error[0].tolist()],
                        [curr_error[1].tolist()],
                        [(1.0, 1.0, 0, 1.0)],
                        [5.0],
                    )

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
