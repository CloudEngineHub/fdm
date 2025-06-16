# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from nav_suite.terrains import NavTerrainImporterCfg

import fdm.mdp as mdp
from fdm import FDM_DATA_DIR

##
# Pre-defined configs
##
# isort: off
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

# NOTE: Uncomment the following imports to enable terrain generation
# from .terrain_cfg import FDM_TERRAINS_CFG
# from .terrain_cfg import FDM_EXTEROCEPTIVE_TERRAINS_CFG
# from .terrain_cfg import PLANNER_TRAIN_CFG
# from .terrain_cfg import MAZE_TERRAIN_CFG

##
# Constants
##

TERRAIN_ANALYSIS_CFG = mdp.TerrainAnalysisCfg(
    semantic_cost_mapping=None,
    raycaster_sensor="env_sensor",
    viz_graph=False,
    viz_height_map=False,
    sample_points=30000,
    height_diff_threshold=0.2,
    wall_height=2.25,
    door_filtering=True,
    grid_resolution=0.05,
    door_height_threshold=1.2,
    max_terrain_size=350.0,
)

##
# Scene definition
##


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # GENERATED TERRAIN
    # terrain = NavTerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     # PILLAR TERRAIN
    #     terrain_generator=FDM_TERRAINS_CFG,
    #     # STAIRS / Stepping Stones / Pillars Terrain
    #     # terrain_generator=FDM_EXTEROCEPTIVE_TERRAINS_CFG,
    #     # PLANNER Stairs/Ramp Terrain
    #     # terrain_generator=PLANNER_TRAIN_CFG,
    #     # Maze Terrain
    #     # terrain_generator=MAZE_TERRAIN_CFG,
    #     max_init_terrain_level=None,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    #     debug_vis=True,
    # )

    # USD TERRAIN
    terrain = NavTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(
            FDM_DATA_DIR, "Terrains", "navigation_terrain_wall_usd_merge_large_single_object_maze.usd"
        ),
        # usd_path=os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_wall_usd_merge_large_maze.usd"),
        # usd_path=os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_wall_usd_merge_large.usd"),
        # usd_path=os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_wall_usd_merge.usd"),
        # usd_path=os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_wall_emptier.usd"),
        # usd_path=os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain.usd"),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
        usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    )

    # robots
    robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5 m above the base for door assessment
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=6, debug_vis=False)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=2000.0,
        ),
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = False


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    command: mdp.NullCommandCfg = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_cmd = mdp.NavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        low_level_decimation=4,
        low_level_policy_file=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Policies", "ANYmal-D", "policy_new.pt"),
        low_level_obs_group="policy",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_cmd"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_cmd"})
        height_scan = ObsTerm(
            func=mdp.height_scan_bounded,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ObsProceptiveCfg(ObsGroup):
        """Propreceptive observations for the FDM."""

        velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_cmd"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_torque = ObsTerm(func=mdp.joint_torque)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel_idx0 = ObsTerm(func=mdp.joint_vel)
        joint_pos_error_idx0 = ObsTerm(func=mdp.joint_pos_error_history, params={"history_idx": 0})
        joint_pos_error_idx2 = ObsTerm(func=mdp.joint_pos_error_history, params={"history_idx": 2})
        joint_pos_error_idx4 = ObsTerm(func=mdp.joint_pos_error_history, params={"history_idx": 4})
        joint_vel_idx2 = ObsTerm(func=mdp.joint_velocity_history, params={"history_idx": 2})
        joint_vel_idx4 = ObsTerm(func=mdp.joint_velocity_history, params={"history_idx": 4})
        last_actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_cmd"})
        second_last_action = ObsTerm(func=mdp.second_last_low_level_action, params={"action_term": "velocity_cmd"})
        # friction = ObsTerm(
        #     func=mdp.friction,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"])},
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class FdmStateCfg(ObsGroup):
        """Observations of the state of the FDM"""

        base_position = ObsTerm(func=mdp.base_position)
        base_orientation = ObsTerm(func=mdp.base_orientation_xyzw)
        base_collision = ObsTerm(
            func=mdp.base_collision,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
            # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "RF_THIGH", "LF_THIGH", "RH_THIGH", "LH_THIGH"]), "threshold": 1.0},
        )
        # add additional terms below, first term have to stay the default and are used in the code
        hard_contact = ObsTerm(func=mdp.energy_consumption, params={"energy_scale_factor": 0.001})
        friction = ObsTerm(
            func=mdp.FrictionObservation(),
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"])},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # FDM observations
    fdm_obs_proprioception: ObsProceptiveCfg = ObsProceptiveCfg()
    fdm_obs_exteroceptive: ObsGroup = None
    # FDM environment observations
    fdm_state: FdmStateCfg = FdmStateCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material_uniform_static_dynamic_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.TerrainAnalysisRootReset(
            cfg=TERRAIN_ANALYSIS_CFG,
            robot_dim=0.6,
        ),
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
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact_delayed,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0, "delay": 1},
        # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "RF_THIGH", "LF_THIGH", "RH_THIGH", "LH_THIGH"]), "threshold": 1.0, "delay": 1},
    )


##
# Environment configuration
##


@configclass
class FDMCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    rerender_on_reset = True
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    # set rewards to None
    rewards = None

    def __post_init__(self):
        """Post initialization."""
        # change ANYmal asset
        # self.scene.robot.spawn.usd_path = f"{FDM_DATA_DIR}/ANYmal-D/anymal_d.usd"
        self.scene.robot.spawn.usd_path = f"{FDM_DATA_DIR}/ANYmal-D-New/anymal_d.usd"
        # set seed
        self.seed = 1234
        # general settings
        self.decimation = 4
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "min"  # important so that the robots are slipping
        self.sim.physics_material.restitution_combine_mode = "min"
        # set render interval correctly
        self.sim.render_interval = self.decimation
        # view settings
        self.viewer.eye = (-5.0, 0, 4)
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
