

from __future__ import annotations

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass

import fdm.mdp as mdp

from .env_cfg_base import FDMCfg, TerrainSceneCfg
from .env_cfg_base_perceptive import PerceptiveFDMCfg, PerceptiveTerrainSceneCfg

##
# Scene definition
##


def modify_scene_cfg(scene_cfg: TerrainSceneCfg):
    # larger height scan
    scene_cfg.env_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.75, 0.0, 5.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(4.5, 5.9)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )


@configclass
class HeightTerrainSceneCfg(TerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __post_init__(self):
        modify_scene_cfg(self)


@configclass
class PerceptiveHeightTerrainSceneCfg(PerceptiveTerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __post_init__(self):
        super().__post_init__()
        modify_scene_cfg(self)


##
# MDP settings
##


@configclass
class ObsExteroceptiveCfg(ObsGroup):
    # collect depth cameras
    env_sensor = ObsTerm(
        func=mdp.height_scan_square, params={"sensor_cfg": SceneEntityCfg("env_sensor"), "shape": (60, 46)}
    )

    def __post_init__(self):
        self.concatenate_terms = True


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
class PerceptiveFDMHeightCfg(PerceptiveFDMCfg):
    """Configuration for the locomotion velocity-tracking environment with perceptive locomotion policy."""

    # Scene settings
    scene: PerceptiveHeightTerrainSceneCfg = PerceptiveHeightTerrainSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False
    )

    def __post_init__(self):
        super().__post_init__()
        # adjust exteroceptive observations
        self.observations.fdm_obs_exteroceptive = ObsExteroceptiveCfg()
