"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument(
    "--env", type=str, default="depth", choices=["lidar", "depth", "height"], help="Name env sensor setting to load."
)
parser.add_argument(
    "--terrain-cfg", type=str, default=None, help="Name of the terrain config to load."
)  # FDM_TERRAINS_CFG  "FDM_EXTEROCEPTIVE_TERRAINS_CFG"
parser.add_argument("--noise", action="store_true", default=False, help="Add noise to the observations.")
parser.add_argument("--friction", action="store_true", default=False, help="Vary friction for each robot.")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
parser.add_argument(
    "--runs", type=str, nargs="+", default=None, help="Name of the run."
)  # ["Dec20_23-58-06_flat_velerr0.0_linear0.4_normal0.3_constant0.3", "Dec21_00-00-17_rough_velerr0.0_linear0.4_normal0.3_constant0.3"]
parser.add_argument(
    "--equal-actions", action="store_true", default=False, help="Have the same actions for all environments."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import fdm.env_cfg as env_cfg
import fdm.mdp as mdp
import fdm.utils.runner_cfg as fdm_runner_cfg
from fdm.utils import FDMRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # setup runner
    if args_cli.env == "lidar":
        cfg = fdm_runner_cfg.RunnerLidarCfg()
    elif args_cli.env == "depth":
        # cfg = fdm_runner_cfg.RunnerDepthCfg()
        cfg = fdm_runner_cfg.RunnerPerceptiveDepthCfg()
    elif args_cli.env == "height":
        cfg = fdm_runner_cfg.RunnerHeightCfg()
    else:
        raise ValueError(f"Unknown environment {args_cli.env}")

    # add noise to observations
    if args_cli.noise:
        cfg.env_cfg.observations.fdm_obs_proprioception.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        cfg.env_cfg.observations.fdm_obs_proprioception.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        cfg.env_cfg.observations.fdm_obs_proprioception.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_torque.noise = Unoise(n_min=-0.1, n_max=0.1)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx0.noise = Unoise(n_min=-1.5, n_max=1.5)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx0.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx2.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx4.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx2.noise = Unoise(n_min=-1.5, n_max=1.5)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx4.noise = Unoise(n_min=-1.5, n_max=1.5)

    # switch to terrain if provided
    if isinstance(args_cli.terrain_cfg, str) and args_cli.terrain_cfg.endswith(".usd"):
        cfg.env_cfg.scene.terrain.usd_path = args_cli.terrain_cfg
    elif isinstance(args_cli.terrain_cfg, str) and hasattr(env_cfg, args_cli.terrain_cfg):
        cfg.env_cfg.scene.terrain.terrain_generator = getattr(env_cfg, args_cli.terrain_cfg)
        cfg.env_cfg.scene.terrain.terrain_generator.num_cols = 2 * len(
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains.items()
        )
        cfg.env_cfg.scene.terrain.terrain_generator.num_rows = 3
        cfg.env_cfg.scene.terrain.terrain_generator.size = (20.0, 20.0)
        cfg.env_cfg.scene.terrain.terrain_generator.curriculum = True
    elif args_cli.terrain_cfg is None:
        print(f"[INFO] Using default terrain config. {cfg.env_cfg.scene.terrain.usd_path}")
    else:
        raise ValueError(f"Unknown terrain config {args_cli.terrain_cfg}")

    # vary friction linearly for each robot
    if args_cli.friction:
        cfg.env_cfg.events.physics_material.func = mdp.regular_rigid_body_material
        cfg.env_cfg.events.physics_material.params["static_friction_range"] = (0.2, 0.8)
        cfg.env_cfg.events.physics_material.params.pop("dynamic_friction_range")
        cfg.env_cfg.events.physics_material.params.pop("restitution_range")
        cfg.env_cfg.events.physics_material.params.pop("num_buckets")

    # set name of the run
    if args_cli.runs is not None:
        cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

    # set regular spawning pattern
    if args_cli.regular:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_regular
        cfg.env_cfg.events.reset_base.params.pop("pose_range")
        cfg.env_cfg.events.reset_base.params.pop("velocity_range")

    # override terrains
    cfg.env_cfg.scene.terrain.usd_path = [
        os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Terrains", "navigation_terrain_stairs.usd"),
        os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Terrains", "navigation_terrain_perlin_stepping_stones.usd"),
        os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Terrains", "navigation_terrain_ramp_platform.usd"),
    ]
    cfg.env_cfg.scene.terrain.usd_translation = (0.0, 0.0, 30.0)

    # add different terrain prims to the sensors
    mesh_prim_paths = [
        cfg.env_cfg.scene.terrain.prim_path + f"/terrain_{idx}"
        for idx in range(len(cfg.env_cfg.scene.terrain.usd_path))
    ]
    if isinstance(cfg, fdm_runner_cfg.RunnerPerceptiveDepthCfg):
        # add to foot scans
        cfg.env_cfg.scene.foot_scanner_lf.mesh_prim_paths = mesh_prim_paths
        cfg.env_cfg.scene.foot_scanner_rf.mesh_prim_paths = mesh_prim_paths
        cfg.env_cfg.scene.foot_scanner_lh.mesh_prim_paths = mesh_prim_paths
        cfg.env_cfg.scene.foot_scanner_rh.mesh_prim_paths = mesh_prim_paths
        # add to env sensors
        cfg.env_cfg.scene.env_sensor.mesh_prim_paths = mesh_prim_paths
        cfg.env_cfg.scene.env_sensor_right.mesh_prim_paths = mesh_prim_paths
        cfg.env_cfg.scene.env_sensor_left.mesh_prim_paths = mesh_prim_paths
    else:
        # add to height scan
        cfg.env_cfg.scene.height_scanner.mesh_prim_paths = mesh_prim_paths
        # add to env sensors
        cfg.env_cfg.scene.env_sensor.mesh_prim_paths = mesh_prim_paths

    # setup runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)

    # eval environments
    runner.eval_env()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
