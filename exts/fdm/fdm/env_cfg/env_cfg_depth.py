# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass

import fdm.mdp as mdp

from .env_cfg_base import FDMCfg, TerrainSceneCfg

##
# Scene definition
##
CAMERA_SIM_RESOLUTION_DECREASE_FACTOR = 6.0


def modify_scene_cfg(scene_cfg: TerrainSceneCfg):
    # sensors
    # frontfacing ZED X Camera
    scene_cfg.env_sensor = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            # pos=(0.4761, 0.0035, 0.1055), rot=(0.9961947, 0.0, 0.087155, 0.0), convention="world"  # 10 degrees
            pos=(0.4761, 0.0035, 0.1055),
            rot=(0.9914449, 0.0, 0.1305262, 0.0),
            convention="world",  # 15 degrees
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        max_distance=10.0,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=10.0,
            horizontal_aperture=20.955,
            height=int(540 / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR),
            width=int(960 / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR),
        ),
    )
    # rearfacing ZED X Camera
    # env_sensor_rear = RayCasterCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     mesh_prim_paths=["/World/ground"],
    #     update_period=0,
    #     offset=RayCasterCameraCfg.OffsetCfg(pos=(-0.4641, 0.0035, 0.1055), rot=(-0.001, 0.132, -0.005, 0.991), convention="world"),  # 10 degrees
    #     data_types=["distance_to_image_plane"],
    #     debug_vis=False,
    #     pattern_cfg=patterns.PinholeCameraPatternCfg(
    #         focal_length=10.0,
    #         horizontal_aperture=20.955,
    #         height=240,
    #         width=320,
    #     ),
    # )
    # left and right facing ZED X Mini Camera
    scene_cfg.env_sensor_right = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0203, -0.1056, 0.1748),
            rot=(0.6963642, 0.1227878, 0.1227878, -0.6963642),
            convention="world",  # 20 degrees
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        max_distance=10.0,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=10.0,
            horizontal_aperture=20.955,
            height=int(540 / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR),
            width=int(960 / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR),
        ),
    )
    scene_cfg.env_sensor_left = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0217, 0.1335, 0.1748),
            rot=(0.6963642, -0.1227878, 0.1227878, 0.6963642),
            convention="world",  # 20 degrees
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        max_distance=10.0,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=10.0,
            horizontal_aperture=20.955,
            height=int(540 / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR),
            width=int(960 / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR),
        ),
    )


@configclass
class DepthTerrainSceneCfg(TerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __post_init__(self):
        modify_scene_cfg(self)


##
# MDP settings
##


@configclass
class ObsExteroceptiveCfg(ObsGroup):
    # collect depth cameras
    right_depth = ObsTerm(
        func=mdp.raycast_depth_camera_data,
        params={"data_type": "distance_to_image_plane", "sensor_cfg": SceneEntityCfg("env_sensor_right")},
    )
    front_depth = ObsTerm(
        func=mdp.raycast_depth_camera_data,
        params={"data_type": "distance_to_image_plane", "sensor_cfg": SceneEntityCfg("env_sensor")},
    )
    left_depth = ObsTerm(
        func=mdp.raycast_depth_camera_data,
        params={"data_type": "distance_to_image_plane", "sensor_cfg": SceneEntityCfg("env_sensor_left")},
    )
    # rear_depth = ObsTerm(
    #     func=mdp.raycast_depth_camera_data,
    #     params={"data_type": "distance_to_image_plane", "sensor_cfg": SceneEntityCfg("env_sensor_rear")},
    # )

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class ObsExteroceptiveHeightScanCfg(ObsGroup):
    # Collect Height Scan Data from the foots
    foot_scan_lf = ObsTerm(
        func=mdp.height_scan_bounded,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_lf"), "offset": 0.05},
        scale=10.0,
    )
    foot_scan_rf = ObsTerm(
        func=mdp.height_scan_bounded,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_rf"), "offset": 0.05},
        scale=10.0,
    )
    foot_scan_lh = ObsTerm(
        func=mdp.height_scan_bounded,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_lh"), "offset": 0.05},
        scale=10.0,
    )
    foot_scan_rh = ObsTerm(
        func=mdp.height_scan_bounded,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_rh"), "offset": 0.05},
        scale=10.0,
    )

    def __post_init__(self):
        self.concatenate_terms = True


##
# Environment configuration
##


@configclass
class FDMDepthCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: DepthTerrainSceneCfg = DepthTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()

        # adjust exteroceptive observations
        self.observations.fdm_obs_exteroceptive = ObsExteroceptiveCfg()
        # add extra exteroceptive observations of the foot surroundings
        self.observations.fdm_add_obs_exteroceptive = ObsExteroceptiveHeightScanCfg()


@configclass
class PreTrainingFDMDepthCfg(FDMDepthCfg):
    """Configuration for the locomotion velocity-tracking environment with perceptive locomotion policy."""

    # Scene settings
    scene: DepthTerrainSceneCfg = DepthTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()

        # add additional height scanner for supervision to the scene
        self.scene.target_height_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(1.75, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(4.5, 5.9)),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            max_distance=10.0,
        )

        # add target height scan to the additional observtiona group
        self.observations.fdm_add_obs_exteroceptive.target_height_scan = ObsTerm(
            func=mdp.height_scan_clipped,
            params={"sensor_cfg": SceneEntityCfg("target_height_scan"), "offset": 0.05},
            clip=(-0.5, 0.5),
        )
