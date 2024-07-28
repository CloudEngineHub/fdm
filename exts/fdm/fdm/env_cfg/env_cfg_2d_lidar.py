

from __future__ import annotations

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils import configclass

import fdm.mdp as mdp
import fdm.sensors.patterns_cfg as patterns_cfg

from .env_cfg_base import FDMCfg, ObservationsCfg, TerrainSceneCfg

##
# Scene definition
##


@configclass
class LidarTerrainSceneCfg(TerrainSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # environment sensor
    env_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),  # TODO: get the correct offset
        pattern_cfg=patterns_cfg.Lidar2DPatternCfg(horizontal_res=1),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )


###
# MDP configuration
###


@configclass
class LidarObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class ObsExteroceptiveCfg(ObsGroup):
        # environment measurements
        env_sensor = ObsTerm(func=mdp.lidar2Dnormalized, params={"sensor_cfg": SceneEntityCfg("env_sensor")})

        def __post_init__(self):
            self.concatenate_terms = True

    # fdm env observations
    fdm_obs_exteroceptive: ObsExteroceptiveCfg = ObsExteroceptiveCfg()


##
# Environment configuration
##


@configclass
class FDMLidarCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: LidarTerrainSceneCfg = LidarTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: LidarObservationsCfg = LidarObservationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
