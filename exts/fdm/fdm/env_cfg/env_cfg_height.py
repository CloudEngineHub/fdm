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

from ..model.fdm_model_cfg import LARGE_UNIFIED_HEIGHT_SCAN
from .env_cfg_base import FDMCfg, TerrainSceneCfg
from .env_cfg_base_mixed import MixedFDMCfg

##
# Scene definition
##


def modify_scene_cfg(scene_cfg: TerrainSceneCfg):
    # larger height scan
    scene_cfg.env_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.75, 0.0, 4.0) if not LARGE_UNIFIED_HEIGHT_SCAN else (0.0, 0.0, 4.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1, size=(4.5, 5.9) if not LARGE_UNIFIED_HEIGHT_SCAN else (7.9, 5.9)
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )


@configclass
class HeightTerrainSceneCfg(TerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __post_init__(self):
        super().__post_init__()
        # change height scan
        modify_scene_cfg(self)


##
# MDP settings
##


@configclass
class ObsExteroceptiveCfg(ObsGroup):
    # collect depth cameras
    env_sensor = ObsTerm(
        func=mdp.height_scan_door_recognition_fdm,
        # func=mdp.height_scan_square_fdm_exp_occlu_with_door_recognition,
        params={
            "sensor_cfg": SceneEntityCfg("env_sensor"),
            "shape": (60, 46) if not LARGE_UNIFIED_HEIGHT_SCAN else (60, 80),
            "offset": 0.5,
        },  # "asset_cfg": SceneEntityCfg("robot"),
        clip=(-1.0, 1.5),
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = True


@configclass
class OccludedObsExteroceptiveCfg(ObsGroup):
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
        # func=mdp.HeightScanOcculusionModifier(
        #     mdp.HeightScanOcculusionModifierCfg(
        #         height_scan_func=mdp.height_scan_square_fdm,
        #         asset_cfg=SceneEntityCfg("robot"),
        #         sensor_cfg=SceneEntityCfg("env_sensor"),
        #         env_ratio=0.5,
        #         sensor_offsets=[[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
        #     )
        # ),
        params={
            "sensor_cfg": SceneEntityCfg("env_sensor"),
            "shape": (60, 46) if not LARGE_UNIFIED_HEIGHT_SCAN else (60, 80),
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
class FDMHeightCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: HeightTerrainSceneCfg = HeightTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        # adjust exteroceptive observations
        self.observations.fdm_obs_exteroceptive = ObsExteroceptiveCfg()


@configclass
class MixedFDMHeightCfg(MixedFDMCfg):
    """Configuration for the locomotion velocity-tracking environment with mixed policy."""

    # Scene settings
    scene: HeightTerrainSceneCfg = HeightTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        # adjust exteroceptive observations
        self.observations.fdm_obs_exteroceptive = ObsExteroceptiveCfg()
