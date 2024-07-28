

from __future__ import annotations

import os

from omni.isaac.lab_assets import ISAACLAB_ASSETS_EXT_DIR

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils import configclass

import fdm.mdp as mdp
import fdm.sensors.patterns_cfg as patterns_cfg

# import base configuration
# isort: off
from .env_cfg_base import FDMCfg, TerrainSceneCfg, ObservationsCfg

ISAAC_GYM_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]


##
# Scene definition
##


@configclass
class PerceptiveTerrainSceneCfg(TerrainSceneCfg):
    foot_scanner_lf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns_cfg.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    foot_scanner_rf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns_cfg.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    foot_scanner_lh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns_cfg.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    foot_scanner_rh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns_cfg.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    def __post_init__(self):
        """Post initialization."""
        self.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        self.height_scanner = None


##
# MDP settings
##


@configclass
class PerceptivePolicyCfg(ObsGroup):
    """Observations for policy group."""

    # Proprioception
    wild_anymal = ObsTerm(
        func=mdp.wild_anymal,
        params={
            "action_term": "velocity_cmd",
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True),
        },
    )
    # Exterocpetion
    foot_scan_lf = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_lf"), "offset": 0.05},
        scale=10.0,
    )
    foot_scan_rf = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_rf"), "offset": 0.05},
        scale=10.0,
    )
    foot_scan_lh = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_lh"), "offset": 0.05},
        scale=10.0,
    )
    foot_scan_rh = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_rh"), "offset": 0.05},
        scale=10.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PerceptiveObsProceptiveCfg(ObservationsCfg.ObsProceptiveCfg):
    """Observations for proprioception group."""

    # add cpg state to the proprioception group
    cpg_state = ObsTerm(func=mdp.cgp_state)

    def __post_init__(self):
        super().__post_init__()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_cmd = mdp.PerceptiveNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        ),
        low_level_decimation=4,
        low_level_policy_file=os.path.join(
            ISAACLAB_ASSETS_EXT_DIR, "Robots/RSL-ETHZ/ANYmal-D", "perceptive_locomotion_jit.pt"
        ),
        reorder_joint_list=ISAAC_GYM_JOINT_NAMES,
    )


##
# Environment configuration
##


@configclass
class PerceptiveFDMCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: PerceptiveTerrainSceneCfg = PerceptiveTerrainSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False
    )
    # Basic settings
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Add policy group
        self.observations.policy = PerceptivePolicyCfg()
        self.observations.fdm_obs_proprioception = PerceptiveObsProceptiveCfg()
