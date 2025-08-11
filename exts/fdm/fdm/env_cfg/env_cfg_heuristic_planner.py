# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import fdm.mdp as mdp

from .env_cfg_base import FDMCfg, TerrainSceneCfg

##
# Scene definition
##


@configclass
class HeuristicsHeightTerrainSceneCfg(TerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    env_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=(7.96, 7.96)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    def __post_init__(self):
        super().__post_init__()


##
# MDP settings
##


@configclass
class HeuristicsObsExteroceptiveCfg(ObsGroup):
    # collect depth cameras
    env_sensor = ObsTerm(
        func=mdp.height_scan_door_recognition_fdm,
        # func=mdp.height_scan_square_fdm_exp_occlu_with_door_recognition,
        params={
            "sensor_cfg": SceneEntityCfg("env_sensor"),
            "shape": (200, 200),
            "offset": 0.5,
        },
        clip=(-1.0, 1.5),
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = True


@configclass
class HeuristicsOccludedObsExteroceptiveCfg(ObsGroup):
    # collect depth cameras
    env_sensor = ObsTerm(
        func=mdp.HeightScanOcculusionDoorRecognitionModifier(
            mdp.HeightScanOcculusionModifierCfg(
                height_scan_func=mdp.height_scan_square_fdm,
                asset_cfg=SceneEntityCfg("robot"),
                sensor_cfg=SceneEntityCfg("env_sensor"),
                env_ratio=0.5,
                sensor_offsets=[[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
            )
        ),
        # func=mdp.HeightScanOcculusionModifierCfg(
        #     height_scan_func=mdp.height_scan_square_fdm,
        #     asset_cfg=SceneEntityCfg("robot"),
        #     sensor_cfg=SceneEntityCfg("env_sensor"),
        #     env_ratio=0.5,
        #     sensor_offsets=[[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
        # ),
        params={
            "sensor_cfg": SceneEntityCfg("env_sensor"),
            "shape": (200, 200),
            "offset": 0.5,
        },
        clip=(-1.0, 1.5),
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = True


##
# Environment configuration
##


@configclass
class FDMHeuristicsHeightCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: HeuristicsHeightTerrainSceneCfg = HeuristicsHeightTerrainSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False
    )

    def __post_init__(self):
        super().__post_init__()
        # adjust exteroceptive observations
        self.observations.fdm_obs_exteroceptive = HeuristicsObsExteroceptiveCfg()
