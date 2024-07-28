

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

import fdm.env_cfg as fdm_env_cfg
import fdm.mdp as mdp
import fdm.model as fdm_model_cfg

from .replay_buffer_cfg import ReplayBufferCfg

# ==
# BASE
# ==


@configclass
class PlannerObsCfg(ObsGroup):
    """Observations for the sampling based planner"""

    goal = ObsTerm(func=mdp.goal_command_w, params={"command_name": "command"})
    start = ObsTerm(func=mdp.se2_root_position)

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class FDMPlannerCfg:
    model_cfg: fdm_model_cfg.FDMBaseModelCfg = MISSING
    """Model config class"""
    env_cfg: fdm_env_cfg.FDMCfg = MISSING
    """Environment config class"""
    replay_buffer_cfg = ReplayBufferCfg()
    """Replay buffer config class"""

    # configurations to load the previous runs
    experiment_name: str = "fdm_se2_prediction_depth"
    """Name of the experiment. """
    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    # path smoothing
    spline_smooth_n: int = 3
    """Number of points to smooth the path with a spline. Default is 3."""

    def __post_init__(self):
        # add planner command
        self.env_cfg.commands.command = mdp.GoalCommandCfg(
            resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
            infite_sampling=False,
            debug_vis=True,
        )

        # add planner observation space
        self.env_cfg.observations.planner_obs = PlannerObsCfg()

        # adjust root state reset based on goal commands spawn positions
        self.env_cfg.events.reset_base = EventTerm(func=mdp.reset_robot_position, mode="reset")
        self.env_cfg.events.reset_robot_joints = None

        # add termination when the goal is reached
        self.env_cfg.terminations.goal_reached = DoneTerm(
            func=mdp.at_goal, params={"distance_threshold": 0.5, "speed_threshold": 1.0}, time_out=True
        )
        # change to non-delayed collision termination bc not interested anymore in recording
        self.env_cfg.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )


# ==
# DEPTH CAMERA BASED PLANNER
# ==


@configclass
class PlannerDepthCfg(FDMPlannerCfg):
    """Configuration for the Planner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMDepthCfg = fdm_env_cfg.FDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthModelCfg = fdm_model_cfg.FDMDepthModelCfg()

    def __post_init__(self):
        super().__post_init__()
        # adjust raycast sensor
        self.env_cfg.commands.command.traj_sampling.terrain_analysis.raycaster_sensor = "height_scanner"


@configclass
class PlannerPerceptiveDepthCfg(PlannerDepthCfg):
    env_cfg: fdm_env_cfg.PerceptiveFDMDepthCfg = fdm_env_cfg.PerceptiveFDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthHeightScanModelCfg = fdm_model_cfg.FDMDepthHeightScanModelCfg()

    def __post_init__(self):
        super().__post_init__()
        # adjust raycast sensor
        self.env_cfg.commands.command.traj_sampling.terrain_analysis.raycaster_sensor = "foot_scanner_rf"

        # debug change to plane
        self.env_cfg.scene.terrain.terrain_type = "plane"
        self.env_cfg.scene.terrain.usd_uniform_env_spacing = None


# ==
# HEIGHT SCAN BASED PLANNER
# ==


@configclass
class PlannerPerceptiveHeightCfg(FDMPlannerCfg):
    """Configuration for the Planner with height scanner data."""

    env_cfg: fdm_env_cfg.PerceptiveFDMHeightCfg = fdm_env_cfg.PerceptiveFDMHeightCfg()
    model_cfg: fdm_model_cfg.FDMHeightModelMultiStepCfg = fdm_model_cfg.FDMHeightModelMultiStepCfg()


@configclass
class PlannerPerceptiveHeightSingleStepCfg(PlannerPerceptiveHeightCfg):
    """Configuration for the Planner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMHeightModelSingleStepCfg = fdm_model_cfg.FDMHeightModelSingleStepCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class PlannerPerceptiveHeightSingleStepHeightAdjustCfg(PlannerPerceptiveHeightCfg):
    """Configuration for the Planner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMModelVelocitySingleStepHeightAdjustCfg = (
        fdm_model_cfg.FDMModelVelocitySingleStepHeightAdjustCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # adjust the height scan pattern
        self.env_cfg.scene.env_sensor.pattern_cfg.resolution = 0.05
        self.env_cfg.scene.env_sensor.pattern_cfg.size = (4.55, 5.95)
        self.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"] = (120, 92)

        # adjust the model parameters
        # TODO: should be able to set them
        # self.model_cfg.height_scan_res = self.env_cfg.scene.env_sensor.pattern_cfg.resolution
        # self.model_cfg.height_scan_shape = list(self.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"])
