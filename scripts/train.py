"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")
parser.add_argument(
    "--env",
    type=str,
    default="height",
    choices=["lidar", "depth", "height", "reduced"],
    help="Name of the environment to load.",
)
parser.add_argument("--noise", action="store_true", default=False, help="Add noise to the observations.")
parser.add_argument("--friction", action="store_true", default=False, help="Vary friction for each robot.")
parser.add_argument("--S4RNN", action="store_true", default=False, help="Use S4RNN instead of GRU.")
parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

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
    # vary friction of each robot
    if args_cli.friction:
        cfg.env_cfg.events.physics_material.params["static_friction_range"] = (0.2, 0.8)
    # init runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli)
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)

    # run
    runner.train()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
