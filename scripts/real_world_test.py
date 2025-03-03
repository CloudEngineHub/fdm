# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test FDM vs perfect velocity model on real world data

The script will load and synchronize the data extracted from the ROSbags and then will the ReplayBuffer. With the filled
ReplayBuffer, the normal evaluation from the FDMRunner can be used.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Fill Replay-Buffer from real-world data.")
# small height scan model:  Nov19_22-48-29_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque_NewHeightScanNoise
# small height new noise:   Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5
# large height scan model:  Nov27_19-52-03_MergeSingleObjMazeTerrain_LargeHeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque_NewHeightScanNoise
parser.add_argument(
    "--runs",
    type=str,
    nargs="+",
    default="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5",
    help="Name of the run.",
)
cli_args.add_fdm_args(parser, default_num_envs=2)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = False

# IMPORTANT: currently developed for the reduced set of observations
args_cli.reduced_obs = True
# IMPORTANT: do not activate noise augmentation to avoid removal of clipping values

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
from dataclasses import MISSING
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation, Slerp

import cv2
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.prims as prim_utils
import kornia
import pypose as pp
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass

from fdm import LARGE_UNIFIED_HEIGHT_SCAN
from fdm.data_buffers import TrajectoryDataset
from fdm.env_cfg.env_cfg_base import CommandsCfg
from fdm.runner import FDMRunner, FDMRunnerCfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, robot_changes, runner_cfg_init
from fdm.utils.colors import generate_colors

# can only be imported if gui activated
try:
    from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
except ImportError:
    omni_debug_draw = None

# joint ordering on the real robot
ANYMAL_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]
REAL_WORLD_ENV_IDX = 0
SIMULATION_ENV_IDX = 1


@configclass
class RealWorldDataCfg:
    export_dir: str = MISSING
    """Path to the directory containing the exported data from the ROSbags"""

    twist_file: str = "twist_mux_twist/data.npy"  # "path_planning_and_following_twist/data.npy"  #
    """Filename of the twist commands"""

    joint_actions_file: str = "anymal_low_level_controller_actuator_readings/data.npy"
    """Filename of the joint actions"""

    state_estimator_file: str = "state_estimator_anymal_state/data.npy"
    """Filename of the state estimator data"""

    elevation_map_file: str = "elevation_mapping_elevation_map_raw/data.npy"
    """Filename of the elevation map"""

    elevation_map_metadata: str = "elevation_mapping_elevation_map_raw/metadata.npy"
    """Filename of the elevation map metadata"""

    odometry_file: str = "gt_box_inertial_explorer_odometry/data.npy"  # "state_estimator_odometry/data.npy" #
    """Filename of the odometry data"""

    goal_point_file: str | None = None  # "clicked_point/data.npy"
    """Filename of the goal point data"""

    path_file: str | None = None  # "planner_node_path/data.npy"
    """Filename of the path data"""

    goal_at_path_time_file: str | None = None  # "planner_node_path/metadata.npy"
    """Filename of the goal at path time data"""

    elevation_map_time_threshold: float = 0.2
    """Threshold in seconds to consider the elevation map as available"""

    ground_truth_odometry: bool = False  # True  #
    """Use the GPS odometry as ground truth for the state.

    If False, will use the odometry from the state estimator. Default is False."""

    show_sim_robot: bool = False
    """Show the simulated robot in the visualization"""


class RealWorldData:
    def __init__(self, cfg: RealWorldDataCfg, runner_cfg: FDMRunnerCfg, args_cli: argparse.Namespace):
        self._cfg = cfg
        self._runner_cfg = runner_cfg
        # setup the runner
        if not self._cfg.show_sim_robot:
            args_cli.num_envs = 1
        self.runner = FDMRunner(cfg=runner_cfg, args_cli=args_cli, render_mode="rgb_array", eval=True)

        # init drawing features
        if omni_debug_draw is not None:
            # init debug draw
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        else:
            raise ImportError("omni_debug_draw not available, please run in GUI mode.")

        # load the data and set constants
        self.GRAVITY_VEC_W = torch.tensor([[0.0, 0.0, -1.0]])
        self._load_data()

        # evaluate the data
        self.eval()

        # play the data with the predictions of the model
        self.play()

    def _load_data(self):
        # load the relevant data
        # -- twist commands
        twist_commands = np.load(os.path.join(self._cfg.export_dir, self._cfg.twist_file))
        # -- joint actions
        joint_actions = np.load(os.path.join(self._cfg.export_dir, self._cfg.joint_actions_file))
        # -- state estimator
        state_estimator = np.load(os.path.join(self._cfg.export_dir, self._cfg.state_estimator_file))
        # -- elevation map
        elevation_map = np.load(os.path.join(self._cfg.export_dir, self._cfg.elevation_map_file))
        elevation_map_metadata = np.load(os.path.join(self._cfg.export_dir, self._cfg.elevation_map_metadata))
        # -- predicted paths
        if self._cfg.path_file is not None and self._cfg.goal_at_path_time_file is not None:
            self.paths = np.load(os.path.join(self._cfg.export_dir, self._cfg.path_file))
            self.goal_at_path_time = np.load(os.path.join(self._cfg.export_dir, self._cfg.goal_at_path_time_file))
        # -- goal point
        if self._cfg.goal_point_file is not None:
            goal_point = np.load(os.path.join(self._cfg.export_dir, self._cfg.goal_point_file))
        # -- odometry
        if self._cfg.ground_truth_odometry:
            odometry = np.load(os.path.join(self._cfg.export_dir, self._cfg.odometry_file))
        else:
            # pose of the base in the world frame (x, y, z, qx, qy, qz, qw) and time (sec, nsec)
            odometry = state_estimator[:, -9:]

        # remove all zero twist commands
        all_zero_twists = np.all(twist_commands[:, :3] == 0.0, axis=1)
        first_non_zero_twist_idx = np.where(~all_zero_twists)[0][0]
        twist_commands = twist_commands[first_non_zero_twist_idx:]

        # -- get the initial time of all the data
        # FIXME: generally start recording once the robot received the first goal and starts planning!!!
        initial_time = np.max([
            joint_actions[0, -2] + joint_actions[0, -1] / 1e9,
            twist_commands[0, -2] + twist_commands[0, -1] / 1e9,
            state_estimator[0, -2] + state_estimator[0, -1] / 1e9,
            elevation_map_metadata[0, -2] + elevation_map_metadata[0, -1] / 1e9,
            odometry[0, -2] + odometry[0, -1] / 1e9,
        ])
        final_time = np.min([
            joint_actions[-1, -2] + joint_actions[-1, -1] / 1e9,
            twist_commands[-1, -2] + twist_commands[-1, -1] / 1e9,
            state_estimator[-1, -2] + state_estimator[-1, -1] / 1e9,
            elevation_map_metadata[-1, -2] + elevation_map_metadata[-1, -1] / 1e9,
            odometry[-1, -2] + odometry[-1, -1] / 1e9,
        ])

        if initial_time > twist_commands[0, -2] + twist_commands[0, -1] / 1e9:
            print("[WARNING] Twist commands start earlier than the other data. Check the data!")

        if final_time < twist_commands[-1, -2] + twist_commands[-1, -1] / 1e9:
            print("[WARNING] Twist commands end later than the other data. Check the data!")

        # synchronize the data
        # NOTE: there are two frequencies, the first one for the proprioceptive data and the second for the exteroceptive
        #       data and twist commands. The synchronization is done for both timestamps and then combined.

        # -- sync proprioceptive data
        #    twist_commands  --> velocity_commands
        #    joint_actions   --> last_actions, second_last_action
        #    state_estimator --> base_lin_vel, base_ang_vel, joint_pos, joint_vel, joint_torque
        #    odometry        --> pos and rot of the base in the world frame
        proprioception_time_step = (
            self._runner_cfg.model_cfg.command_timestep / self._runner_cfg.model_cfg.history_length
        )
        recording_timestamps = np.arange(initial_time, final_time + proprioception_time_step, proprioception_time_step)

        # -- find slowest data
        #    twist commands are not necessary, as they are repeated in between the recordings
        joint_actions_freq = 1 / np.mean(np.diff(joint_actions[:, -2] + joint_actions[:, -1] / 1e9))
        state_estimator_freq = 1 / np.mean(np.diff(state_estimator[:, -2] + state_estimator[:, -1] / 1e9))
        odometry_freq = 1 / np.mean(np.diff(odometry[:, -2] + odometry[:, -1] / 1e9))
        # -- check that it is at minimum equal to the proprioception frequency used during training
        assert (
            min(joint_actions_freq, state_estimator_freq) >= 1 / proprioception_time_step
        ), "The slowest data is slower than the training frequency."

        # check if the odometry is slower than the training frequency, then interpolate the odometry
        if odometry_freq < 1 / proprioception_time_step:
            odometry = self.interpolate_poses(odometry, int(odometry_freq), int(state_estimator_freq))

        odometry_idx = self._sync_data(odometry[:, -2] + odometry[:, -1] / 1e9, recording_timestamps)
        joint_action_idx = self._sync_data(joint_actions[:, -2] + joint_actions[:, -1] / 1e9, recording_timestamps)
        state_estimator_idx = self._sync_data(
            state_estimator[:, -2] + state_estimator[:, -1] / 1e9, recording_timestamps
        )
        twist_commands_idx = self._sync_data(twist_commands[:, -2] + twist_commands[:, -1] / 1e9, recording_timestamps)

        # -- check elevation map frequency
        elevation_map_freq = 1 / np.mean(np.diff(elevation_map_metadata[:, -2] + elevation_map_metadata[:, -1] / 1e9))
        assert (
            elevation_map_freq >= 1 / self._runner_cfg.model_cfg.command_timestep
        ), "The elevation map is slower than the training frequency."

        # -- sync exteroceptive data
        elevation_map_idx = self._sync_data(
            elevation_map_metadata[:, -2] + elevation_map_metadata[:, -1] / 1e9, recording_timestamps
        )

        # NOTE: elevation map is not available at any point of time, mark at which ones to govern the sampling of the trajectory dataset
        command_recording_timestamps = np.arange(
            initial_time,
            final_time - self._runner_cfg.model_cfg.command_timestep,
            self._runner_cfg.model_cfg.command_timestep,
        )
        elevation_map_idx_command_timestamp = self._sync_data(
            elevation_map_metadata[:, -2] + elevation_map_metadata[:, -1] / 1e9, command_recording_timestamps
        )
        self.elevation_map_available = (elevation_map_metadata[:, -2] + elevation_map_metadata[:, -1] / 1e9)[
            elevation_map_idx_command_timestamp
        ] - command_recording_timestamps < self._cfg.elevation_map_time_threshold
        self.elevation_map_available = torch.tensor(self.elevation_map_available)

        # save the predicted paths on the robot and the goal positions for plotting
        # -- sync the paths with each timestep
        if self._cfg.path_file is not None and self._cfg.goal_at_path_time_file is not None:
            self.paths_idx = self._sync_data(
                self.goal_at_path_time[:, -2] + self.goal_at_path_time[:, -1] / 1e9, command_recording_timestamps
            )
        if self._cfg.goal_point_file is not None:
            self.goal_idx = self._sync_data(goal_point[:, -2] + goal_point[:, -1] / 1e9, command_recording_timestamps)

        # move data to torch
        twist_commands = torch.tensor(twist_commands).to(torch.float32)
        joint_actions = torch.tensor(joint_actions).to(torch.float32)
        state_estimator = torch.tensor(state_estimator).to(torch.float32)
        elevation_map = torch.tensor(elevation_map).to(torch.float32)
        elevation_map_metadata = torch.tensor(elevation_map_metadata).to(torch.float32)
        odometry = torch.tensor(odometry).to(torch.float32)

        # construct the joint mapping between the real-world and the simulation
        joint_mapping = torch.tensor([
            ANYMAL_JOINT_NAMES.index(joint_name)
            for joint_name in self.runner.env.scene.articulations["robot"].joint_names
        ])

        # for the twist commands, check if a skipped command is an interpolation between the current and the next recorded one
        twist_command_idx_diff = np.diff(twist_commands_idx)
        twist_command_with_intermediate = twist_commands_idx[:-1][twist_command_idx_diff > 1]
        twist_command_num_intermediate = twist_command_idx_diff[twist_command_idx_diff > 1]
        for idx, curr_intermediate in enumerate(twist_command_with_intermediate):
            command_mean = (
                twist_commands[curr_intermediate + twist_command_num_intermediate[idx], :3]
                + twist_commands[curr_intermediate, :3]
            ) / 2
            if any([
                torch.any(torch.abs(twist_commands[curr_intermediate + i + 1, :3] - command_mean) > 0.1)
                for i in range(twist_command_num_intermediate[idx])
            ]):
                # print(f"Twist command {curr_intermediate} has a skipped command with a too large difference.")
                # print(f"Start: {twist_commands[curr_intermediate, :3]}")
                # print(f"End: {twist_commands[curr_intermediate + twist_command_num_intermediate[idx], :3]}")
                # print(f"Intermediates: {twist_commands[curr_intermediate + 1 : curr_intermediate + twist_command_num_intermediate[idx], :3]}")
                # print(f"Diff: {twist_commands[curr_intermediate + twist_command_num_intermediate[idx], :3] - twist_commands[curr_intermediate, :3]}")
                # print("-----------------")

                # replace start with mean of start and intermediate
                twist_commands[curr_intermediate, :3] = (
                    twist_commands[curr_intermediate, :3] + twist_commands[curr_intermediate + 1, :3]
                ) / 2

        # construct the information
        # -- state: (x, y, z, qx, qy, qz, qw, collision=0, energy)
        #        odometry: x, y, z, qx, qy, qz, qw
        #        collision: 0   -- don't have collision on the real robot
        #        energy: sum(state_estimator[24:36] ** 2) * energy_scale_factor
        #        friction: [0, 0, 0, 0] --> not used for the network input but loss for pre-training (still included in state)

        # hard contact value
        if self._runner_cfg.model_cfg.hard_contact_metric == "energy":
            hard_contact_value = (
                torch.sum(state_estimator[state_estimator_idx, 24:36] ** 2, dim=1).unsqueeze(1)
                * self._runner_cfg.env_cfg.observations.fdm_state.hard_contact.params["energy_scale_factor"]
                - self.runner.model.hard_contact_obs_limits[0].to("cpu")
            ) / (
                self.runner.model.hard_contact_obs_limits[1].to("cpu")
                - self.runner.model.hard_contact_obs_limits[0].to("cpu")
            )
        else:
            raise NotImplementedError("Only energy based hard contact metric is supported.")
            hard_contact_value = torch.zeros((len(recording_timestamps), 1), dtype=torch.float32)

        # get the states
        states = torch.concatenate(
            [
                # -- odometry: pos and rotation
                odometry[odometry_idx, :7],
                # -- collision: all zeros bc not in collision
                torch.zeros((len(recording_timestamps), 1), dtype=torch.float32),
                # -- hard contact estimation value
                hard_contact_value,
                # -- friction (all zero, as it will be ignored later)
                torch.zeros((len(recording_timestamps), 4)),
            ],
            dim=1,
        )
        # -- observations_proprioceptive: (velocity_commands, projected_gravity, base_lin_vel, base_ang_vel, joint_torque, joint_pos, joint_vel_idx0, last_actions, second_last_action)
        observations_proprioceptive = torch.concatenate(
            [
                # [0, 2]   twist commands  --> vx, vy, wz
                twist_commands[twist_commands_idx, :3],
                # [3, 5]   projected_gravity --> gx, gy, gz
                math_utils.quat_rotate_inverse(
                    odometry[odometry_idx][:, [6, 3, 4, 5]], self.GRAVITY_VEC_W.repeat(len(odometry_idx), 1)
                ),
                # [6, 11]  state_estimator --> base_lin_vel (36:39), base_ang_vel (39:42)
                state_estimator[state_estimator_idx, 36:42],
                # [12, 23] state estimator --> joint_torque (24:36)
                (
                    state_estimator[state_estimator_idx, 24:36][:, joint_mapping]
                    if not args_cli.remove_torque
                    else torch.tensor([])
                ),
                # [24, 35] state estimator --> joint_pos (0:12)
                state_estimator[state_estimator_idx, :12][:, joint_mapping],
                # [36, 47] state estimator --> joint_vel_idx0 (12:24)
                state_estimator[state_estimator_idx, 12:24][:, joint_mapping],
                # [48, 59] last actions
                joint_actions[joint_action_idx, :12][:, joint_mapping],
                # [60, 71] second last action
                joint_actions[joint_action_idx - 1, :12][:, joint_mapping],
            ],
            dim=1,
        )
        observations_proprioceptive[0, -12:] = 0  # set the second last action to zero for the first step
        # -- observations_exteroceptive: (elevation_map)

        # -- transform the sample points into the current odom frame
        sample_points = math_utils.quat_apply_yaw(
            math_utils.quat_inv(
                odometry[odometry_idx, 3:7][:, [3, 0, 1, 2]]
                .unsqueeze(1)
                .repeat(1, self.runner.env.scene.sensors["env_sensor"].ray_starts.shape[1], 1)
                .view(-1, 4)
            ),
            self.runner.env.scene.sensors["env_sensor"]
            .ray_starts[0]
            .to("cpu")
            .unsqueeze(0)
            .repeat(len(odometry_idx), 1, 1)
            .view(-1, 3),
        ).reshape(len(odometry_idx), self.runner.env.scene.sensors["env_sensor"].ray_starts.shape[1], 3)
        sample_points += (
            odometry[odometry_idx, :3] - elevation_map_metadata[elevation_map_idx, :3].to("cpu")
        ).unsqueeze(1)
        # -- convert to idx by diving through the resolution of the elevation map
        sample_idx = torch.round(sample_points[..., :2] / elevation_map_metadata[0, -3]).to(torch.int)
        # -- indexes assume middle as center, correct for it
        sample_idx += (torch.tensor(elevation_map.shape[1:3]) - 1) // 2
        # -- clip the indexes to the elevation map size
        sample_idx = torch.clip(sample_idx, 0, elevation_map.shape[1] - 1)
        # -- need to rotate the elevation map by 90 degrees
        elevation_map = torch.rot90(elevation_map, 1, dims=(1, 2))
        # -- get the elevation map at the sample points
        observations_exteroceptive = elevation_map[
            torch.tensor(elevation_map_idx)[:, None].repeat(1, sample_idx.shape[1]),
            sample_idx[..., 0],
            sample_idx[..., 1],
        ]
        observations_exteroceptive = torch.unflatten(
            observations_exteroceptive,
            1,
            self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"],
        )
        # has to flip the image
        observations_exteroceptive = torch.flip(observations_exteroceptive, dims=[1])

        if False:
            import matplotlib.pyplot as plt

            plt.imshow(elevation_map[elevation_map_idx[0]])
            plt.scatter(sample_idx[0, :, 1].to("cpu"), sample_idx[0, :, 0].to("cpu"), color="red", s=2)

        # observation exteroceptie is still in odom frame, has to be elevated to the common frmae of the sensor box
        # in the case the sensor box is not used, this will not do anything
        observations_exteroceptive += elevation_map_metadata[elevation_map_idx, 2].to("cpu").unsqueeze(1).unsqueeze(1)

        # repeat the same operations for the height map as done in sim --> transfer to
        observations_exteroceptive += self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.params[
            "offset"
        ] - odometry[odometry_idx, 2].to("cpu").unsqueeze(1).unsqueeze(1)

        # set the unobserved elevation map values
        observations_exteroceptive[torch.isnan(observations_exteroceptive)] = self.runner.env.scene.sensors[
            "env_sensor"
        ].cfg.max_distance

        # apply an median filter to the height map
        # -- pad by the size of the median filter
        padding = (2, 2, 2, 2)
        observations_exteroceptive = F.pad(observations_exteroceptive, padding, mode="replicate")
        # -- apply filter
        observations_exteroceptive = kornia.filters.median_blur(
            observations_exteroceptive.unsqueeze(1), (5, 5)
        ).squeeze(1)
        # -- remove padding
        observations_exteroceptive = observations_exteroceptive[:, 2:-2, 2:-2]

        # either clip or make nearest neighbor filling
        if self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.noise is not None:
            mask = (
                observations_exteroceptive
                > self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.noise.occlusion_height
            )
            indices = np.zeros((len(observations_exteroceptive.shape), *mask.shape), dtype=np.int32)

            # convert to nunmpy
            mask_np = mask.cpu().numpy()

            # Compute the distance transform and nearest neighbor indices
            for i in range(mask.shape[0]):
                indices[0, i] = i
                _, indices[1:, i] = distance_transform_edt(mask_np[i], return_indices=True)

            # Use indices to assign nearest neighbor values
            observations_exteroceptive = observations_exteroceptive[tuple(indices)]
        else:
            # clip to max observable height
            observations_exteroceptive = torch.clip(
                observations_exteroceptive,
                self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.clip[0],
                self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.clip[1],
            )

        # debug
        if False:
            # visualize 10 height maps
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 5)
            for i in range(5):
                # choose a random height map
                idx = np.random.randint(900, 1200)
                # plot the height map
                axs[0, i % 5].imshow(observations_exteroceptive[idx].cpu().numpy())
                axs[1, i % 5].imshow(elevation_map[elevation_map_idx][idx].cpu().numpy())
                # add title with the index
                axs[0, i % 5].set_title(f"idx: {idx}")
            # save fig
            plt.savefig("/home/pascal/elevation_map.png")
            plt.close()

        # update the length of the replay buffer and reinitialize it
        self.runner.replay_buffer.cfg.trajectory_length = len(command_recording_timestamps)
        self.runner.replay_buffer._init_buffers()

        # fill the replay buffer
        for step_idx in range(len(recording_timestamps) - 1):
            # check that environment is not filled yet
            if self.runner.replay_buffer.is_filled:
                print(f"Replay buffer filled after {step_idx} of {len(recording_timestamps)} steps.")
                break

            # add to replay buffer
            self.runner.replay_buffer.add(
                states=states[step_idx].unsqueeze(0).repeat(args_cli.num_envs, 1),
                obersevations_proprioceptive=observations_proprioceptive[step_idx]
                .unsqueeze(0)
                .repeat(args_cli.num_envs, 1),
                obersevations_exteroceptive=observations_exteroceptive[step_idx]
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(args_cli.num_envs, 1, 1, 1),
                # FIXME: future actions or the current ones?  --> is the anymal msg the current applied ones?  CHECK ONES DATA RECORDING WITH PLANNER
                actions=observations_proprioceptive[step_idx + 1, :3].repeat(args_cli.num_envs, 1),
                dones=torch.zeros(args_cli.num_envs, dtype=torch.bool),
                feet_contact=torch.ones(args_cli.num_envs, dtype=torch.bool),
                add_observation_exteroceptive=None,
            )

            # manually increase the step counter bc data is already in the timesteps of the proprioceptive recording
            env_step = (
                self.runner.replay_buffer.env_step_counter[0]
                + self.runner.replay_buffer.history_collection_interval
                - 1
            )
            if (torch.floor(env_step) % self.runner.replay_buffer.history_collection_interval).to(torch.int) == 0:
                self.runner.replay_buffer.env_step_counter[:] = torch.floor(env_step)
            else:
                self.runner.replay_buffer.env_step_counter[:] = torch.ceil(env_step)

    def eval(self):
        # fill trajectory dataset
        # NOTE: sample only at points where a new height map is available and the horizon is not exceeded
        self.elevation_map_available[(-self.runner.cfg.model_cfg.prediction_horizon - 1) :] = False
        sample_idx = torch.nonzero(self.elevation_map_available).squeeze(1).unsqueeze(0)
        start_idx = torch.concatenate([torch.zeros_like(sample_idx), sample_idx], dim=0).T
        self.initial_state = self.runner.trainer.train_dataset.populate(
            replay_buffer=self.runner.replay_buffer, start_idx=start_idx
        )[0]

        # evaluate the model
        mean_eval_loss = 0
        batch_number = len(self.runner.trainer.dataloader)
        for batch_idx, inputs in enumerate(self.runner.trainer.dataloader):
            loss, meta = self.runner.model.evaluate(model_in=inputs[:5], target=inputs[5], eval_in=inputs[6:])
            mean_eval_loss += loss

            if batch_idx == 0:
                meta_eval = meta
            else:
                for key, value in meta.items():
                    try:
                        meta_eval[key] += value
                    except KeyError:
                        meta_eval[key] = value

        # average meta_eval
        for key, value in meta_eval.items():
            meta_eval[key] = value / batch_number
        mean_eval_loss = mean_eval_loss / batch_number
        # print meta information
        self.runner.trainer.print_meta_info(meta_eval)

        # split the dataset into train and val set, save both to start training with real-world data
        train_dataset = TrajectoryDataset(
            cfg=self.runner.trainer.cfg,
            replay_buffer_cfg=self.runner.cfg.replay_buffer_cfg,
            model_cfg=self.runner.model.cfg,
            return_device=self.runner.device,
        )
        val_dataset = TrajectoryDataset(
            cfg=self.runner.trainer.cfg,
            replay_buffer_cfg=self.runner.cfg.replay_buffer_cfg,
            model_cfg=self.runner.model.cfg,
            return_device=self.runner.device,
        )
        # create random indices for the split into train and val dataset
        split_idx = torch.randperm(len(start_idx))
        train_dataset.populate(self.runner.replay_buffer, start_idx=start_idx[split_idx[: int(len(start_idx) * 0.8)]])
        val_dataset.populate(self.runner.replay_buffer, start_idx=start_idx[split_idx[int(len(start_idx) * 0.8) :]])

        # save the datasets
        dataset_path = os.path.join(self._cfg.export_dir, "real_world_dataset")
        os.makedirs(dataset_path, exist_ok=True)
        file_name = "real_world_dataset"
        if args_cli.reduced_obs:
            file_name += "_reducedObs"
        if args_cli.env == "baseline":
            file_name += "_baseline"
        if LARGE_UNIFIED_HEIGHT_SCAN:
            file_name += "_largeUnifiedHeightScan"
        if args_cli.remove_torque:
            file_name += "_noTorque"
        if args_cli.noise:
            file_name += "_nearest_neighbor_filling"
        with open(os.path.join(dataset_path, file_name + "_train.pkl"), "wb") as fp:
            pickle.dump(train_dataset, fp)
            print(f"[INFO] Train dataset saved to {dataset_path}/{file_name}_train.pkl")
        with open(os.path.join(dataset_path, file_name + "_val.pkl"), "wb") as fp:
            pickle.dump(val_dataset, fp)
            print(f"[INFO] Val dataset saved to {dataset_path}/{file_name}_val.pkl")

    def play(self):
        """Play the collected data with the model predictions and the compared model in simulation.

        We visualize:
        - the collected robot state (base and joint pos and rotation)
        - the collected height scan measurement
        - the FDM predictions
        - the perfect velocity model predictions
        - a simulated robot with the same actions
        """
        real_world_env_idx = torch.tensor([REAL_WORLD_ENV_IDX], dtype=torch.int32)
        simulation_env_idx = torch.tensor([SIMULATION_ENV_IDX], dtype=torch.int32)
        nb_draw_traj = 2

        safe_colors = generate_colors(nb_draw_traj, start_hue=0.3, end_hue=0.4)  # green
        collision_colors = generate_colors(nb_draw_traj, start_hue=0.0, end_hue=0.05)  # red
        prediction_color_real = [(1.0, 0.65, 0.0, 1.0)]  # orange
        real_goal_color = [(1.0, 0.85, 0.0, 1.0)]  # gold
        trajectory_color_real = [(0.0, 0.0, 1.0, 1.0)]  # blue
        trajectory_color_sim = [(0.2, 0.9, 0.8, 1.0)]  # turquoise
        perfect_velocity_color = [(1.0, 0.6, 0.0, 1.0)]  # purple
        future_trajectory_color = [(0.5, 0.0, 1.0, 1.0)]  # vilett

        # elevate the states/ elevation map in the replay buffer to be over the ground
        # if self.runner.replay_buffer.states[0, 0, 0, 2] < 0.6:
        #     ground_diff = 0.6 - self.runner.replay_buffer.states[0, 0, 0, 2]
        #     self.runner.replay_buffer.states[..., 2] += ground_diff
        #     self.runner.replay_buffer.observations_exteroceptive += ground_diff

        # reset the environment
        with torch.inference_mode():
            obs, _ = self.runner.env.reset(1)

        # set the plan visualization to false
        ground_prim = prim_utils.get_prim_at_path(
            os.path.join(self.runner.env.scene.terrain.cfg.prim_path, "terrain", "Environment", "Geometry")
        )
        ground_prim.GetAttribute("visibility").Set("false")

        # set the root position of the robots to the starting position
        self.runner.env.scene.articulations["robot"].write_root_pose_to_sim(
            self.runner.replay_buffer.states[..., 0, 0, [0, 1, 2, 6, 3, 4, 5]],
        )

        # create output directory for the visualization images
        rgb_output_dir = os.path.join(self._cfg.export_dir, self.runner.trainer.cfg.load_run, "renders")
        os.makedirs(rgb_output_dir, exist_ok=True)

        # get predictions
        with torch.inference_mode():
            pred_state, pred_coll, pred_eng = self.runner.model.forward((
                self.runner.trainer.train_dataset.state_history,
                self.runner.trainer.train_dataset.obs_proprioceptive,
                self.runner.trainer.train_dataset.obs_exteroceptive.to(torch.float32),
                self.runner.trainer.train_dataset.actions,
                torch.zeros(self.runner.trainer.train_dataset.actions.shape[0]),
                self.runner.trainer.train_dataset.states,
                self.runner.trainer.train_dataset.perfect_velocity_following_local_frame,
            ))
            pred_state[..., 2] = torch.atan2(pred_state[..., 2], pred_state[..., 3])
            pred_state = pred_state[..., :3].reshape(-1, 3).cpu()

            perf_vel_pred = self.runner.trainer.train_dataset.perfect_velocity_following_local_frame.clone()
            perf_vel_pred[..., 2] = torch.atan2(perf_vel_pred[..., 2], perf_vel_pred[..., 3])
            perf_vel_pred = perf_vel_pred[..., :3].reshape(-1, 3).cpu()

        # transform predictions into global frame
        pred_state_SE3 = pp.SE3(
            torch.concatenate(
                [
                    pred_state[..., :2],
                    torch.zeros((pred_state.shape[0], 1)),
                    math_utils.quat_from_euler_xyz(
                        torch.zeros_like(pred_state[:, 2]), torch.zeros_like(pred_state[:, 2]), pred_state[:, 2]
                    )[:, [1, 2, 3, 0]],
                ],
                dim=1,
            )
        )
        pred_state_SE3_odom = (pp.SE3(self.initial_state.reshape(-1, 7)) * pred_state_SE3).tensor()
        pred_state_plot = torch.concatenate(
            [
                pred_state_SE3_odom[:, :2],
                math_utils.euler_xyz_from_quat(pred_state_SE3_odom[:, [6, 3, 4, 5]])[2].unsqueeze(-1),
            ],
            dim=1,
        )
        pred_state_plot = pred_state_plot.reshape(-1, self.runner.model.cfg.prediction_horizon, 3)
        pred_state_plot[..., 2] = torch.concatenate(
            [
                self.runner.replay_buffer.states[
                    real_world_env_idx, (1 + i) : -(self.runner.model.cfg.prediction_horizon - i), 0, 2
                ]
                .squeeze(0)
                .unsqueeze(1)
                for i in range(self.runner.model.cfg.prediction_horizon)
            ],
            dim=-1,
        )

        perf_vel_pred_SE3 = pp.SE3(
            torch.concatenate(
                [
                    perf_vel_pred[..., :2],
                    torch.zeros((perf_vel_pred.shape[0], 1)),
                    math_utils.quat_from_euler_xyz(
                        torch.zeros_like(perf_vel_pred[:, 2]),
                        torch.zeros_like(perf_vel_pred[:, 2]),
                        perf_vel_pred[:, 2],
                    )[:, [1, 2, 3, 0]],
                ],
                dim=1,
            )
        )
        perf_vel_pred_SE3_odom = (pp.SE3(self.initial_state.reshape(-1, 7)) * perf_vel_pred_SE3).tensor()
        perf_vel_pred_plot = torch.concatenate(
            [
                perf_vel_pred_SE3_odom[:, :2],
                math_utils.euler_xyz_from_quat(perf_vel_pred_SE3_odom[:, [6, 3, 4, 5]])[2].unsqueeze(-1),
            ],
            dim=1,
        )
        perf_vel_pred_plot = perf_vel_pred_plot.reshape(-1, self.runner.model.cfg.prediction_horizon, 3)
        perf_vel_pred_plot[..., 2] = torch.concatenate(
            [
                self.runner.replay_buffer.states[
                    real_world_env_idx, (1 + i) : -(self.runner.model.cfg.prediction_horizon - i), 0, 2
                ]
                .squeeze(0)
                .unsqueeze(1)
                for i in range(self.runner.model.cfg.prediction_horizon)
            ],
            dim=-1,
        )

        # initialize the visualization markers for the true raycast scan
        ray_visualizer = VisualizationMarkers(
            self._runner_cfg.env_cfg.scene.env_sensor.visualizer_cfg.replace(prim_path="/Visuals/RayCasterReal")
        )
        ray_visualizer.set_visibility(True)

        # play the simulation
        sim_pos_buffer = torch.zeros((self.runner.replay_buffer.states.shape[1], 3), device=self.runner.device)
        nb_substeps = int(
            self.runner.model.cfg.command_timestep / (self.runner.env.physics_dt * self.runner.env.cfg.decimation)
        )

        # init buffers for energy plotting
        sim_energy_buffer = torch.zeros(
            (self.runner.replay_buffer.states.shape[1] * nb_substeps, 1), device=self.runner.device
        )

        for command_idx in range(self.runner.replay_buffer.states.shape[1] - 1):

            # step the simulator with the same actions for the prediction time
            history_idx = torch.zeros(1, dtype=torch.int)
            for substep_idx in range(nb_substeps):
                # step the simulation
                with torch.inference_mode():
                    if self._cfg.show_sim_robot:
                        obs = self.runner.env.step(self.runner.replay_buffer.actions[:, command_idx])[0]
                    else:
                        self.runner.env.render()
                    # save rgb render
                    assert cv2.imwrite(
                        os.path.join(
                            rgb_output_dir, f"img_{str(command_idx * nb_substeps + substep_idx).zfill(4)}.png"
                        ),
                        cv2.cvtColor(self.runner.env.render(), cv2.COLOR_RGB2BGR),
                    )

                    # update the current history idx depending on the amount of substeps done (negative because selected from the next command)
                    if (
                        int(substep_idx % self.runner.replay_buffer.history_collection_interval) == 0
                        and substep_idx != 0
                    ):  # noqa: E721
                        history_idx -= 1

                    # when progressing, change to the recording for the next command
                    if history_idx == 0:
                        selected_command_idx = command_idx
                    else:
                        selected_command_idx = command_idx + 1

                    # override the base and joint position and orientation of the real-world data robot
                    self.runner.env.scene.articulations["robot"].write_root_pose_to_sim(
                        self.runner.replay_buffer.states[
                            real_world_env_idx, selected_command_idx, history_idx, [0, 1, 2, 6, 3, 4, 5]
                        ].to(self.runner.env.device),
                        env_ids=real_world_env_idx.to(self.runner.env.device),
                    )
                    if not args_cli.remove_torque:
                        joint_pos_idx = [24, 36]
                        joint_vel_idx = [36, 48]
                    else:
                        joint_pos_idx = [12, 24]
                        joint_vel_idx = [24, 36]

                    self.runner.env.scene.articulations["robot"].write_joint_state_to_sim(
                        position=self.runner.replay_buffer.observations_proprioceptive[
                            real_world_env_idx, selected_command_idx, history_idx, joint_pos_idx[0] : joint_pos_idx[1]
                        ].to(self.runner.env.device),
                        velocity=self.runner.replay_buffer.observations_proprioceptive[
                            real_world_env_idx, selected_command_idx, history_idx, joint_vel_idx[0] : joint_vel_idx[1]
                        ].to(self.runner.env.device),
                        env_ids=real_world_env_idx.to(self.runner.env.device),
                    )

                    # save energy consumption
                    if self._cfg.show_sim_robot:
                        sim_energy_buffer[command_idx * nb_substeps + substep_idx] = obs["fdm_state"][
                            simulation_env_idx, -5
                        ]

            # override the observed height scan
            # TODO: check if the height scan is in the correct shape after the reshaping operation
            if not self._cfg.show_sim_robot:
                self.runner.env.scene.sensors["env_sensor"].update(self.runner.env.step_dt)
            ray_hits = self.runner.env.scene.sensors["env_sensor"].data.ray_hits_w[real_world_env_idx].clone()
            ray_hits[..., 2] = (
                self.runner.replay_buffer.observations_exteroceptive[real_world_env_idx, command_idx].flatten(1)
                + self.runner.replay_buffer.states[0, command_idx, 0, 2]
                - self.runner.env.cfg.observations.fdm_obs_exteroceptive.env_sensor.params["offset"]
            )
            ray_visualizer.visualize(ray_hits.view(-1, 3))

            if torch.any(torch.isnan(ray_hits)):
                raise ValueError("NaN in the ray hits.")

            # update sim pos buffer
            if self._cfg.show_sim_robot:
                sim_pos_buffer[command_idx] = obs["fdm_state"][simulation_env_idx, :3]

            # only execute trajectory drawing if walked a bit
            if command_idx == 0:
                continue

            # render the FDM predictions as well as the trajectories of both robots
            self.draw_interface.clear_lines()

            # draw the future n steps of the real-world robot
            max_command_idx = min(
                command_idx + self.runner.model.cfg.prediction_horizon, self.runner.replay_buffer.states.shape[1] - 1
            )
            self.draw_interface.draw_lines(
                self.runner.replay_buffer.states[real_world_env_idx, command_idx : max_command_idx - 1, 0, :3]
                .view(-1, 3)
                .tolist(),
                self.runner.replay_buffer.states[real_world_env_idx, command_idx + 1 : max_command_idx, 0, :3]
                .view(-1, 3)
                .tolist(),
                future_trajectory_color * (max_command_idx - command_idx - 1),
                [5.0] * (max_command_idx - command_idx - 1),
            )
            # plot trajectories from replay buffer
            min_command_idx = max(0, command_idx - self.runner.model.cfg.prediction_horizon)
            self.draw_interface.draw_lines(
                self.runner.replay_buffer.states[real_world_env_idx, min_command_idx:command_idx, 0, :3]
                .view(-1, 3)
                .tolist(),
                self.runner.replay_buffer.states[real_world_env_idx, min_command_idx + 1 : command_idx + 1, 0, :3]
                .view(-1, 3)
                .tolist(),
                trajectory_color_real * (command_idx - min_command_idx),
                [5.0] * (command_idx - min_command_idx),
            )
            # draw simulated trajectory
            if self._cfg.show_sim_robot:
                self.draw_interface.draw_lines(
                    sim_pos_buffer[min_command_idx:command_idx].tolist(),
                    sim_pos_buffer[min_command_idx + 1 : command_idx + 1].tolist(),
                    trajectory_color_sim * (command_idx - min_command_idx),
                    [5.0] * (command_idx - min_command_idx),
                )
            # define number of trajectories to draw
            if (
                command_idx
                > self.runner.replay_buffer.states.shape[1] - 1 - self.runner.cfg.model_cfg.prediction_horizon
            ):
                command_idx_applied = (
                    self.runner.replay_buffer.states.shape[1] - 1 - self.runner.cfg.model_cfg.prediction_horizon
                )
                start_idx = command_idx_applied - nb_draw_traj
            elif command_idx > nb_draw_traj:
                command_idx_applied = command_idx
                start_idx = command_idx - nb_draw_traj
            else:
                command_idx_applied = command_idx
                start_idx = 0

            # get the colors
            colors = []
            for i in range(min(command_idx_applied, nb_draw_traj)):
                if torch.any(pred_coll[start_idx + i] > self._runner_cfg.model_cfg.collision_threshold):
                    colors.extend([collision_colors[i]] * (self.runner.model.cfg.prediction_horizon - 1))
                else:
                    colors.extend([safe_colors[i]] * (self.runner.model.cfg.prediction_horizon - 1))

            self.draw_interface.draw_lines(
                pred_state_plot[start_idx:command_idx_applied, :-1].reshape(-1, 3).tolist(),
                pred_state_plot[start_idx:command_idx_applied, 1:].reshape(-1, 3).tolist(),
                colors,
                [5.0] * (min(command_idx_applied, nb_draw_traj) * (self.runner.model.cfg.prediction_horizon - 1)),
            )
            # draw perfect velocity model predictions
            self.draw_interface.draw_lines(
                perf_vel_pred_plot[start_idx:command_idx_applied, :-1].reshape(-1, 3).tolist(),
                perf_vel_pred_plot[start_idx:command_idx_applied, 1:].reshape(-1, 3).tolist(),
                [perfect_velocity_color[0]]
                * (min(command_idx_applied, nb_draw_traj) * (self.runner.model.cfg.prediction_horizon - 1)),
                [5.0] * (min(command_idx_applied, nb_draw_traj) * (self.runner.model.cfg.prediction_horizon - 1)),
            )

            # plot the goal point and the predicted paths on the robot if available
            if self._cfg.path_file is not None:
                paths = self.paths[self.paths_idx[command_idx_applied], :, :3]
                paths[..., 2] = 0.5
                self.draw_interface.draw_lines(
                    paths[:-1].tolist(),
                    paths[1:].tolist(),
                    prediction_color_real * (len(paths) - 1),
                    [5.0] * (len(paths) - 1),
                )
            if self._cfg.goal_at_path_time_file is not None:
                goal = self.goal_at_path_time[self.goal_idx[start_idx:command_idx_applied], :3]
                goal[..., 2] = 0.5
                self.draw_interface.draw_points(
                    goal.tolist(),
                    real_goal_color,
                    [15.0],
                )

            print(f"Step {command_idx} of {self.runner.replay_buffer.states.shape[1] - 1}")

        # plot the used, simulated and predicted energy
        import matplotlib.pyplot as plt

        # scale the predicted energy
        pred_eng = (
            pred_eng * (self.runner.model.hard_contact_obs_limits[1] - self.runner.model.hard_contact_obs_limits[0])
            + self.runner.model.hard_contact_obs_limits[0]
        ).to("cpu")

        for pred_idx in range(self.runner.model.cfg.prediction_horizon):
            idx_commands = np.arange(
                1 + pred_idx,
                (self.runner.replay_buffer.states.shape[1] - (self.runner.model.cfg.prediction_horizon - pred_idx))
                * nb_substeps,
                nb_substeps,
            )[: self.runner.replay_buffer.states.shape[1] - self.runner.model.cfg.prediction_horizon - 1]
            plt.plot(
                idx_commands,
                self.runner.replay_buffer.states[
                    REAL_WORLD_ENV_IDX, (1 + pred_idx) : -(self.runner.model.cfg.prediction_horizon - pred_idx), 0, -5
                ]
                .cpu()
                .numpy(),
                label="Real",
            )
            if self._cfg.show_sim_robot:
                plt.plot(
                    np.arange(
                        1 + pred_idx,
                        (
                            self.runner.replay_buffer.states.shape[1]
                            - (self.runner.model.cfg.prediction_horizon - pred_idx)
                        )
                        * nb_substeps,
                    ),
                    sim_energy_buffer[
                        (1 + pred_idx) : -(self.runner.model.cfg.prediction_horizon - pred_idx) * nb_substeps
                    ]
                    .cpu()
                    .numpy(),
                    label="Simulated",
                )
            plt.plot(idx_commands, pred_eng[:, pred_idx].cpu().numpy(), label="Predicted")
            plt.title(f"Energy consumption over time for prediction step {pred_idx}")
            plt.xlabel("Time step")
            plt.ylabel("Energy consumption")
            plt.legend()
            plt.savefig(
                os.path.join(self._cfg.export_dir, self.runner.trainer.cfg.load_run, f"energy_pred_{pred_idx}.png")
            )
            plt.close()

        # put the visualization into a gif
        output_file_name = "output.mp4" if not args_cli.noise else "output_nearest_neighbor_filling.mp4"
        save_file = os.path.join(self._cfg.export_dir, self.runner.trainer.cfg.load_run, output_file_name)
        os.system(
            f"ffmpeg -r {int(1 / self.runner.env.step_dt)} -f image2 -s 1920x1080 -i"
            f" '{rgb_output_dir}/img_%04d.png' -vcodec libx264 -profile:v high -crf 25 -pix_fmt yuv420p"
            f" '{save_file}'"
        )

    """
    Helper functions
    """

    def _plot_odometry(self, positions, original_positions: np.ndarray | None = None):
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot original and interpolated positions
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "o-", label="Positions", markersize=5)
        if original_positions is not None:
            ax.plot(
                original_positions[:, 0],
                original_positions[:, 1],
                original_positions[:, 2],
                "-",
                label="Original",
                alpha=0.8,
            )

        # Labels and legend
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Positions (3D)")
        ax.legend()

        # Save the plot to a file
        dataset_path = os.path.join(self._cfg.export_dir, "real_world_dataset")
        file_path = os.path.join(dataset_path, "odometry.png")
        os.makedirs(dataset_path, exist_ok=True)
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved as '{file_path}'")

    @staticmethod
    def interpolate_poses(data, original_frequency: int = 20, target_frequency: int = 400):
        """
        Interpolates poses from lower frequency to target_frequency Hz.

        Parameters:
            data (numpy.ndarray): Array of shape (N, 9) with columns [x, y, z, qx, qy, qz, qw, t_sec, t_nsec].
            target_frequency (int): Desired output frequency. Default is 400Hz.

        Returns:
            numpy.ndarray: Interpolated data with the same columns as input.
        """
        interpolation_factor = target_frequency // original_frequency

        # Extract components
        positions = data[:, :3]
        quaternions = data[:, 3:7]
        timestamps_sec = data[:, 7]
        timestamps_nsec = data[:, 8]
        timestamps = timestamps_sec + timestamps_nsec * 1e-9

        # Generate original and target time arrays
        original_time = np.linspace(0, timestamps[-1] - timestamps[0], len(timestamps))
        target_time = np.linspace(0, timestamps[-1] - timestamps[0], len(timestamps) * interpolation_factor)

        # Linear interpolation for positions and timestamps
        inter_func = interp1d(original_time, positions.T)
        interpolated_positions = inter_func(target_time).T

        # Handle quaternion interpolation using SLERP
        rotations = Rotation.from_quat(quaternions)
        slerp = Slerp(original_time, rotations)
        interpolated_rotations = slerp(target_time)
        interpolated_quaternions = interpolated_rotations.as_quat()

        # Interpolated timestamps
        interpolated_timestamps = np.interp(target_time, original_time, timestamps)

        # Convert interpolated timestamps back to secs and nsecs
        interpolated_t_sec = np.floor(interpolated_timestamps).astype(int)
        interpolated_t_nsec = ((interpolated_timestamps - interpolated_t_sec) * 1e9).astype(int)

        # Combine results
        interpolated_data = np.hstack((
            interpolated_positions,
            interpolated_quaternions,
            interpolated_t_sec[:, None],
            interpolated_t_nsec[:, None],
        ))

        return interpolated_data

    @staticmethod
    def _sync_data(data_timestamps: np.ndarray, recording_timestamps: np.ndarray):
        """Get for each recording timestamp the index of the data timestamp clostest to the recording but timewise larger."""
        # sort recording timestamps into the data timestamps
        # NOTE: this means insert_rec_into_data[i] is the index where recording timestamp i is inserted into the data
        #       timestamps s.t. it is smaller than data_timestamps[insert_rec_into_data[i]] and larger than
        #       data_timestamps[insert_rec_into_data[i] - 1]
        # NOTE: the -1 is necessary to get the index of the data timestamp that is timewise larger than the recording timestamp
        return np.clip(np.searchsorted(data_timestamps, recording_timestamps), 0, len(data_timestamps) - 1)


def main():
    # init runner cfg
    args_cli.env = "height"
    runner_cfg = runner_cfg_init(args_cli)
    # select robot
    runner_cfg = robot_changes(runner_cfg, args_cli)
    # modify the runner cfg
    runner_cfg = cfg_modifier_pre_init(runner_cfg, args_cli=args_cli)

    # increase the height of the height scanner to make sure it always hits the ground plane
    pos_list = list(runner_cfg.env_cfg.scene.env_sensor.offset.pos)
    pos_list[2] = 100.0
    runner_cfg.env_cfg.scene.env_sensor.offset.pos = tuple(pos_list)
    runner_cfg.env_cfg.scene.env_sensor.max_distance = 150.0

    # set the run to the one that should be loaded
    if args_cli.runs is not None:
        runner_cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

    # change environment to plane and deactivate
    runner_cfg.env_cfg.scene.terrain.terrain_type = "plane"
    runner_cfg.env_cfg.events.reset_base = None
    runner_cfg.env_cfg.commands = CommandsCfg()
    runner_cfg.env_cfg.observations.planner_obs = None
    runner_cfg.agent_cfg = None

    # overwrite some configs for easier debugging
    runner_cfg.trainer_cfg.logging = False
    runner_cfg.trainer_cfg.test_datasets = None
    runner_cfg.trainer_cfg.small_motion_ratio = None

    # set the viewer config to follow the env robot
    runner_cfg.env_cfg.viewer.origin_type = "asset_root"
    runner_cfg.env_cfg.viewer.asset_name = "robot"
    runner_cfg.env_cfg.viewer.env_index = REAL_WORLD_ENV_IDX

    # filter non-moving robot data from the extracted dataset
    runner_cfg.trainer_cfg.small_motion_ratio = 0.0
    runner_cfg.trainer_cfg.small_motion_threshold = 0.1

    # GrandTour DataSets with GPS odometry
    # real_world_cfg = RealWorldDataCfg(
    #     # export_dir="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-07-52-45_moenchsjoch_fenced_1/export",
    #     # export_dir="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-07-57-34_moenchsjoch_fenced_2/export",
    #     export_dir="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-08-17-23_moenchsjoch_outside_1/export",
    #     # export_dir="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-08-42-30_moenchsjoch_outside_2/export",
    #     odometry_file="gt_box_inertial_explorer_tc_odometry/data.npy",
    #     ground_truth_odometry=True,
    # )

    # GrandTour DataSets with DLIO odometry
    real_world_cfg = RealWorldDataCfg(
        # export_dir="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-14-14-36-02_forest_kaeferberg_entanglement/export",
        export_dir=(
            "/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-15-12-06-03_forest_albisguetli_slippery_slope/export"
        ),
        odometry_file="dlio_lidar_map_odometry/data.npy",
        ground_truth_odometry=True,
    )

    # Polyterasse DataSets
    # real_world_cfg = RealWorldDataCfg(
    #     export_dir="/media/pascal/T7 Shield/FDMData/2024-09-23-Polyterasse/2024-09-23-10-52-57/fdm_relevant/export",
    #     odometry_file="gt_box_inertial_explorer_odometry/data.npy",
    #     ground_truth_odometry=True,
    # )

    # LeggedOdom DataSets
    # real_world_cfg = RealWorldDataCfg(
    #     export_dir="/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-13-11_stairs/fdm_relevant/export",
    #     # export_dir="/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-21-14_meeting_room/fdm_relevant/export",
    # )

    RealWorldData(real_world_cfg, runner_cfg, args_cli)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
