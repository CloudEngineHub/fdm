# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils import configclass

import fdm.mdp as mdp
import fdm.sensors.patterns_cfg as patterns

from .env_cfg_base import FDMCfg, TerrainSceneCfg
from .terrain_cfg import BASELINE_2D_TERRAIN_CFG, FDM_TERRAINS_CFG  # noqa: F401

##
# Scene definition
##


@configclass
class BaselineTerrainSceneCfg(TerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # environment sensor
    env_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),  # TODO: get the correct offset
        pattern_cfg=patterns.Lidar2DPatternCfg(horizontal_res=1),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        max_distance=4.0,
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # change the terrain to 2D case
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = BASELINE_2D_TERRAIN_CFG  # FDM_TERRAINS_CFG


###
# MDP configuration
###


@configclass
class BaselineObsProprioceptiveCfg(ObsGroup):
    """Propreceptive observations for the FDM."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class BaselineFdmStateCfg(ObsGroup):
    """Observations of the state of the FDM"""

    base_position = ObsTerm(func=mdp.base_position)
    base_orientation = ObsTerm(func=mdp.base_orientation_xyzw)
    base_collision = ObsTerm(
        func=mdp.base_collision,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
        # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "RF_THIGH", "LF_THIGH", "RH_THIGH", "LH_THIGH"]), "threshold": 1.0},
    )
    # add additional terms below, first term have to stay the default and are used in the code
    # NOTE: this is required in the trajectory dataset code but will not be used by the Baseline model
    hard_contact = ObsTerm(func=mdp.energy_consumption, params={"energy_scale_factor": 0.001})

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class ObsExteroceptiveCfg(ObsGroup):
    # environment measurements
    env_sensor = ObsTerm(func=mdp.lidar2Dnormalized, params={"sensor_cfg": SceneEntityCfg("env_sensor")})

    def __post_init__(self):
        self.concatenate_terms = True


##
# Environment configuration
##


@configclass
class FDMBaselineEnvCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: BaselineTerrainSceneCfg = BaselineTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # override observation groups
        self.observations.fdm_obs_exteroceptive = ObsExteroceptiveCfg()
        self.observations.fdm_obs_proprioception = BaselineObsProprioceptiveCfg()
        self.observations.fdm_state = BaselineFdmStateCfg()
