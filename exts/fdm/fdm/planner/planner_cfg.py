# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from nav_suite.collectors import TrajectorySamplingCfg

import fdm.env_cfg as fdm_env_cfg
import fdm.mdp as mdp
import fdm.model as fdm_model_cfg
from fdm.env_cfg.ui.planner_ui_window import PlannerEnvWindow

# ==
# BASE
# ==


@configclass
class PlannerObsCfg(ObsGroup):
    """Observations for the sampling based planner"""

    goal = ObsTerm(func=mdp.goal_command_w_se2, params={"command_name": "command"})
    start = ObsTerm(func=mdp.se2_root_position)

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class FDMPlannerCfg:
    model_cfg: fdm_model_cfg.FDMBaseModelCfg = MISSING
    """Model config class"""
    env_cfg: fdm_env_cfg.FDMCfg = MISSING
    """Environment config class"""

    # configurations to load the previous runs
    experiment_name: str = "fdm_se2_prediction_depth"
    """Name of the experiment. """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model.*.pt"
    """The checkpoint file to load. Default is "model.*.pt" (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    spline_smooth_n: int = 3
    """Number of points to smooth the path with a spline. Default is 3."""

    frequency: int = 10
    """Frequency of the planner. Default is 10."""

    movement_threshold: float = 0.2
    """Threshold to be considered moving. Default is 0.2.

    If the norm of the applied velocity command is below the threshold, the robot is considered to be stopped.
    After :attr:`movement_resample_count` consecutive steps of being stopped, the planner will resample the population.
    """

    movement_resample_count: int = 30
    """Number of consecutive steps to be considered stopped. Default is 30.

    Will resample the population if the robot is stopped for this many steps."""

    max_path_time: float = 20.0
    """Maximum time to come to the goal in sec. Default is 15.0."""

    def __post_init__(self):
        # set the correct length of the max torque buffer if the term exists
        if not isinstance(self.env_cfg, type(MISSING)) and self.model_cfg.hard_contact_metric == "torque":
            self.env_cfg.observations.fdm_state.hard_contact = mdp.MaxJointTorqueCfg(history_length=10)
            self.env_cfg.observations.fdm_state.hard_contact.history_length = math.ceil(
                self.model_cfg.command_timestep
                / (self.env_cfg.sim.dt * self.env_cfg.decimation * self.model_cfg.history_length)
            )
        if not isinstance(self.env_cfg, type(MISSING)) and self.model_cfg.hard_contact_metric == "contact":
            self.env_cfg.observations.fdm_state.hard_contact = mdp.MaxContactForceObsCfg(
                history_length=10,
                params={
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
                    )
                },
            )
            self.env_cfg.observations.fdm_state.hard_contact.history_length = math.ceil(
                self.model_cfg.command_timestep
                / (self.env_cfg.sim.dt * self.env_cfg.decimation * self.model_cfg.history_length)
            )

        # add planner command
        self.env_cfg.commands.command = mdp.GoalCommandCfg(
            resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
            sampling_mode="bounded",
            debug_vis=True,
            traj_sampling=TrajectorySamplingCfg(
                terrain_analysis=fdm_env_cfg.TERRAIN_ANALYSIS_CFG,
            ),
        )
        # disable saved paths loading if terrains are randomly generated
        if self.env_cfg.scene.terrain.terrain_type == "generator":
            self.env_cfg.commands.command.traj_sampling.enable_saved_paths_loading = False

        # add planner observation space
        self.env_cfg.observations.planner_obs = PlannerObsCfg()

        # adjust root state reset based on goal commands spawn positions
        self.env_cfg.events.reset_base = EventTerm(
            func=mdp.reset_robot_position_planner,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "yaw_range": (-3.14, 3.14),
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0, 0),
                    "roll": (0, 0),
                    "pitch": (0, 0),
                    "yaw": (-0.5, 0.5),
                },
                "goal_command_generator_name": "command",
            },
        )
        self.env_cfg.events.reset_robot_joints = None

        # add termination when the goal is reached
        self.env_cfg.terminations.goal_reached = DoneTerm(
            func=mdp.at_goal,
            params={"distance_threshold": 0.5, "speed_threshold": 1.0, "command_generator_term_name": "command"},
            time_out=True,
        )
        # change to non-delayed collision termination bc not interested anymore in recording
        self.env_cfg.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )
        self.env_cfg.scene.contact_forces.history_length = 3
        # add a timeout termination
        # NOTE: has timeout set to false to classify the run as unsuccessful
        self.env_cfg.terminations.timeout = DoneTerm(func=mdp.time_out, time_out=False)
        self.env_cfg.episode_length_s = 20.0

        # add ui window for planning
        self.env_cfg.ui_window_class_type = PlannerEnvWindow

        # make environment origins regular
        self.env_cfg.scene.terrain.regular_spawning = True


# ==
# Baseline
# ==


@configclass
class PlannerBaselineCfg(FDMPlannerCfg):
    """Configuration of the Planner using the Baseline FDM network."""

    env_cfg: fdm_env_cfg.FDMBaselineEnvCfg = fdm_env_cfg.FDMBaselineEnvCfg()
    model_cfg: fdm_model_cfg.FDMBaselineCfg = fdm_model_cfg.FDMBaselineCfg()

    def __post_init__(self):
        super().__post_init__()

        # adjust the experiment name
        self.experiment_name = "fdm_baseline"


# ==
# DEPTH CAMERA BASED PLANNER
# ==


@configclass
class PlannerDepthCfg(FDMPlannerCfg):
    """Configuration for the Planner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMDepthCfg = fdm_env_cfg.FDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthModelCfg = fdm_model_cfg.FDMDepthModelCfg()


# ==
# Ours: HEIGHT SCAN BASED PLANNER
# ==


@configclass
class PlannerHeightCfg(FDMPlannerCfg):
    """Configuration for the Planner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMHeightCfg = fdm_env_cfg.FDMHeightCfg()
    model_cfg: fdm_model_cfg.FDMHeightModelMultiStepCfg = fdm_model_cfg.FDMHeightModelMultiStepCfg()


@configclass
class PlannerHeightSingleStepCfg(PlannerHeightCfg):
    """Configuration for the Planner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMHeightModelSingleStepCfg = fdm_model_cfg.FDMHeightModelSingleStepCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class PlannerHeightSingleStepHeightAdjustCfg(PlannerHeightCfg):
    """Configuration for the Planner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMModelVelocitySingleStepHeightAdjustCfg = (
        fdm_model_cfg.FDMModelVelocitySingleStepHeightAdjustCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        raise NotImplementedError("This configuration is not yet implemented.")

        # adjust the height scan pattern
        self.env_cfg.scene.env_sensor.pattern_cfg.resolution = 0.05
        self.env_cfg.scene.env_sensor.pattern_cfg.size = (4.55, 5.95)
        self.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"] = (120, 92)


# ==
# Comparisons Planner
# ==


@configclass
class PlannerHeuristicCfg(FDMPlannerCfg):
    """Configuration for the Planner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMHeuristicsHeightCfg = fdm_env_cfg.FDMHeuristicsHeightCfg()
    model_cfg: fdm_model_cfg.FDMHeightModelMultiStepCfg = fdm_model_cfg.FDMHeightModelMultiStepCfg()
    # NOTE: technicall the model config is not necessary, however, the config is used at some parts in the code
