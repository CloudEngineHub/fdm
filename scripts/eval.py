"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate.")
parser.add_argument(
    "--env", type=str, default="height", choices=["lidar", "depth", "height"], help="Name env sensor setting to load."
)
parser.add_argument("--terrain-cfg", type=str, default=None, help="Name of the terrain config to load.")
parser.add_argument("--noise", action="store_true", default=False, help="Add noise to the observations.")
parser.add_argument("--S4RNN", action="store_true", default=False, help="Use S4RNN instead of GRU.")
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

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import fdm.env_cfg as env_cfg
import fdm.mdp as mdp
from fdm.utils import FDMRunner
from fdm.utils.args_cli_utils import (
    cfg_modifier_pre_init,
    env_modifier_post_init,
    runner_cfg_init,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # init runner cfg
    cfg = runner_cfg_init(args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # overwrite some configs for easier debugging
    cfg.replay_buffer_cfg.trajectory_length = 50
    cfg.trainer_cfg.num_samples = 2000
    cfg.trainer_cfg.logging = False

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

    # setup runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)
    # set initial state and initial predictions, plot them
    runner.evaluate()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
