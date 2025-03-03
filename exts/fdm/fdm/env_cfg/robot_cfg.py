# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import nav_tasks.sensors as nav_patterns
from nav_tasks import NAVSUITE_TASKS_DATA_DIR
from nav_tasks.mdp.actions.navigation_se2_actions_cfg import ISAAC_GYM_JOINT_NAMES

import fdm.mdp as mdp

from .env_cfg_base import FDMCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.tytan import TYTAN_CFG  # isort: skip
from isaaclab_assets.robots.aow import ANYMAL_C_ON_WHEELS_CFG  # isort: skip


###
# ANYmal Perceptive
###


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
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_cmd = mdp.PerceptiveNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        ),
        low_level_decimation=4,
        low_level_policy_file=os.path.join(NAVSUITE_TASKS_DATA_DIR, "Policies", "perceptive_locomotion_jit.pt"),
        low_level_obs_group="policy",
    )


def anymal_perceptive(cfg: FDMCfg) -> FDMCfg:
    """Apply changes to the FDM configuration for the ANYmal Perceptive environment."""
    # change height scanner for the robot
    cfg.scene.height_scanner = None
    cfg.scene.foot_scanner_lf = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        attach_yaw_only=True,
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    cfg.scene.foot_scanner_rf = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        attach_yaw_only=True,
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    cfg.scene.foot_scanner_lh = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        attach_yaw_only=True,
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )

    cfg.scene.foot_scanner_rh = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        attach_yaw_only=True,
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
    )
    # change default position
    cfg.scene.robot.init_state.joint_pos = {
        "LF_HAA": -0.13859,
        "LH_HAA": -0.13859,
        "RF_HAA": 0.13859,
        "RH_HAA": 0.13859,
        ".*F_HFE": 0.480936,  # both front HFE
        ".*H_HFE": -0.480936,  # both hind HFE
        ".*F_KFE": -0.761428,
        ".*H_KFE": 0.761428,
    }
    # add cpg state to observation state of the policy
    cfg.observations.fdm_obs_proprioception.cpg_state = ObsTerm(func=mdp.cgp_state)
    # change the low level contron actions
    cfg.actions = ActionsCfg()
    # change policy observation
    cfg.observations.policy = PerceptivePolicyCfg()

    # change raycast sensor for goal command sampling if commands in the environment
    if hasattr(cfg.commands.command, "traj_sampling"):
        cfg.commands.command.traj_sampling.terrain_analysis.raycaster_sensor = "foot_scanner_rf"

    return cfg


###
# Tytan
###


@configclass
class TytanPolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_cmd"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
    )
    actions = ObsTerm(
        func=mdp.last_low_level_action,
        params={
            "action_term": "velocity_cmd",
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def tytan_env(cfg: FDMCfg, quiet: bool = False) -> FDMCfg:
    """Apply changes to the FDM configuration for the Tytan environment."""
    # change robot in the scene
    cfg.scene.robot = TYTAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # adapt the action config
    cfg.actions.velocity_cmd.low_level_action.scale = 0.25
    cfg.actions.velocity_cmd.low_level_action.joint_names = [".*HAA", ".*HFE", ".*KFE"]
    cfg.actions.velocity_cmd.low_level_action.use_default_offset = True
    if quiet:
        cfg.actions.velocity_cmd.low_level_policy_file = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Policies/RSL-ETHZ/Tytan/quiet_policy.pt"
        )
    else:
        cfg.actions.velocity_cmd.low_level_policy_file = os.path.join(
            ISAACLAB_ASSETS_DATA_DIR, "Policies/RSL-ETHZ/Tytan/policy.pt"
        )
    cfg.actions.velocity_cmd.reorder_joint_list = ISAAC_GYM_JOINT_NAMES
    # change policy observation
    cfg.observations.policy = TytanPolicyCfg()
    # remove all history obersvations as not compatible with the AOW actuators
    cfg.observations.fdm_obs_proprioception.joint_pos_error_idx0 = None
    cfg.observations.fdm_obs_proprioception.joint_pos_error_idx2 = None
    cfg.observations.fdm_obs_proprioception.joint_pos_error_idx4 = None
    cfg.observations.fdm_obs_proprioception.joint_vel_idx2 = None
    cfg.observations.fdm_obs_proprioception.joint_vel_idx4 = None
    # swap friction body names to the shank in the fdm state observation
    cfg.observations.fdm_state.friction.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=["LF_SHANK", "LH_SHANK", "RF_SHANK", "RH_SHANK"]
    )
    # swap friction body names to the shank in the randomization as startup time
    cfg.events.physics_material.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=["LF_SHANK", "LH_SHANK", "RF_SHANK", "RH_SHANK"]
    )

    # change the height of the height scanner of the policy to be able to catch doors
    cfg.scene.height_scanner.offset.pos = (0.0, 0.0, 0.5)

    return cfg


###
# AOW
###

AOW_ISAAC_GYM_ACTUATOR_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LF_WHEEL",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "LH_WHEEL",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RF_WHEEL",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
    "RH_WHEEL",
]


@configclass
class AoWPolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_cmd"}, scale=(3.0, 3.0, 0.25))
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=AOW_ISAAC_GYM_ACTUATOR_NAMES, preserve_order=True)
        },
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_low_level_action,
        params={
            "action_term": "velocity_cmd",
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=AOW_ISAAC_GYM_ACTUATOR_NAMES, preserve_order=True),
        },
    )
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        clip=(-1.0, 1.0),
        scale=5.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def aow_env(cfg: FDMCfg, env: str) -> FDMCfg:
    """Apply changes to the FDM configuration for the AOW environment."""
    # change robot in the scene
    cfg.scene.robot = ANYMAL_C_ON_WHEELS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # add necessary height scanner
    cfg.scene.height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # change the low level contron actions
    cfg.actions.velocity_cmd.low_level_action = [
        mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*HAA", ".*HFE", ".*KFE"], scale=0.5, use_default_offset=True
        ),
        mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*WHEEL"], scale=5.0, use_default_offset=True),
    ]
    cfg.actions.velocity_cmd.reorder_joint_list = AOW_ISAAC_GYM_ACTUATOR_NAMES
    cfg.actions.velocity_cmd.low_level_policy_file = os.path.join(
        ISAACLAB_ASSETS_DATA_DIR, "Policies/RSL-ETHZ/AoW/policy_walking.pt"
    )
    # change policy observation
    cfg.observations.policy = AoWPolicyCfg()
    # remove all history obersvations as not compatible with the AOW actuators
    cfg.observations.fdm_obs_proprioception.joint_pos_error_idx0 = None
    cfg.observations.fdm_obs_proprioception.joint_pos_error_idx2 = None
    cfg.observations.fdm_obs_proprioception.joint_pos_error_idx4 = None
    cfg.observations.fdm_obs_proprioception.joint_vel_idx2 = None
    cfg.observations.fdm_obs_proprioception.joint_vel_idx4 = None
    # swap friction body names to the wheels in the fdm state observation
    cfg.observations.fdm_state.friction.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=["LF_WHEEL_L", "LH_WHEEL_L", "RF_WHEEL_L", "RH_WHEEL_L"]
    )
    # swap friction body names to the wheels in the randomization as startup time
    cfg.events.physics_material.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=["LF_WHEEL_L", "LH_WHEEL_L", "RF_WHEEL_L", "RH_WHEEL_L"]
    )

    # adjust the env height scanner, as the AoW is faster than ANYmal (if env is height)
    if env == "height":
        # adjust the sensor
        cfg.scene.env_sensor.offset.pos = (2.25, 0.0, 4.0)
        cfg.scene.env_sensor.pattern_cfg.size = (6.4, 5.9)
        cfg.scene.env_sensor.pattern_cfg.resolution = 0.1

        # adjust the observation terms for the new shape
        cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"] = (60, 65)

    return cfg
