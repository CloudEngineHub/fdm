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
parser.add_argument("--noise", action="store_true", default=False, help="Add noise to the observations.")
parser.add_argument("--friction", action="store_true", default=False, help="Vary friction for each robot.")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
parser.add_argument("--run", type=str, default=None, help="Name of the run.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import fdm.mdp as mdp
import fdm.utils.planner_cfg as fdm_planner_cfg
from fdm.utils import FDMPlanner
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # setup runner
    if args_cli.env == "depth":
        cfg = fdm_planner_cfg.PlannerPerceptiveDepthCfg()
    elif args_cli.env == "height":
        cfg = fdm_planner_cfg.PlannerPerceptiveHeightCfg()
        # cfg = fdm_planner_cfg.PlannerPerceptiveHeightSingleStepCfg()
        # cfg = fdm_planner_cfg.PlannerPerceptiveHeightSingleStepHeightAdjustCfg()
    else:
        raise ValueError(f"Unknown/ Not yet supported environment {args_cli.env}")

    # DEBUG
    args_cli.num_envs = 2
    cfg.env_cfg.scene.terrain.terrain_type = "plane"
    cfg.env_cfg.scene.terrain.usd_uniform_env_spacing = False

    # follow robot
    # cfg.env_cfg.viewer.asset_name = "robot"
    # cfg.env_cfg.viewer.origin_type = "asset_root"

    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # vary friction linearly for each robot
    if args_cli.friction:
        cfg.env_cfg.events.physics_material.func = mdp.regular_rigid_body_material
        cfg.env_cfg.events.physics_material.params["static_friction_range"] = (0.2, 0.8)
        cfg.env_cfg.events.physics_material.params.pop("dynamic_friction_range")
        cfg.env_cfg.events.physics_material.params.pop("restitution_range")
        cfg.env_cfg.events.physics_material.params.pop("num_buckets")

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # set regular spawning pattern
    if args_cli.regular:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_regular
        cfg.env_cfg.events.reset_base.params.pop("pose_range")
        cfg.env_cfg.events.reset_base.params.pop("velocity_range")

    debug = False
    device = "cuda"
    traj_dim = 25

    cfg_dict = {
        "traj_dim": traj_dim,
        "action_cfg": {
            "_target_": "fdm.sampling_planner.ActionCfg",
            "action_dim": 3,
            "traj_dim": traj_dim,
            "lower_bound": [-1.2, -0.8, -torch.pi],
            "upper_bound": [1.2, 0.8, torch.pi],
        },
        "to_cfg": {
            "_target_": "fdm.sampling_planner.TrajectoryOptimizerCfg",
            "init_debug": debug,
            "debug": debug,
            "dt": 0.20,
            "n_step_fwd": True,
            "control": "velocity_control",
            "state_cost_w_early_goal_reaching": 0,
            "state_cost_w_action_trans_forward": 0,
            "state_cost_w_action_trans_side": 60,
            "state_cost_w_action_rot": 30,
            "state_cost_w_fatal_unknown": 500,
            "state_cost_w_fatal_trav": 1000,
            "state_cost_w_risky_unknown": 0,
            "state_cost_w_risky_trav": 0,
            "state_cost_w_cautious_unknown": 0,
            "state_cost_w_cautious_trav": 0,
            "state_cost_velocity_tracking": 0,
            "state_cost_desired_velocity": 1.0,
            "state_cost_early_goal_distance_offset": 0.3,
            "state_cost_early_goal_heading_offset": 100,
            "terminal_cost_w_rot_error": 20,
            "terminal_cost_w_position_error": 60,
            "terminal_cost_distance_offset": 0.3,
            "terminal_cost_close_reward": 500,
            "terminal_cost_use_threshold": True,
            "pp_safe_th": 0.3,
            "pp_risky_th": 0.4,
            "pp_fatal_th": 0.8,
            "pp_risky_value": 0.5,
            "pp_fatal_value": 1,
        },
        "optim": {
            "_target_": "fdm.sampling_planner.BatchedMPPIOptimizer",
            "num_iterations": 3,
            "population_size": 1024,
            "gamma": 0.91,
            "sigma": 0.87,
            "beta": 0.08,
            "lower_bound": ["${action_cfg.lower_bound}" for i in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for i in range(traj_dim)],
            "device": device,
            "batch_size": args_cli.num_envs,
        },
        # "optim": {
        #     "_target_": "fdm.sampling_planner.BatchedICEMOptimizer",
        #     "num_iterations": 5,
        #     "elite_ratio": 0.03,
        #     "alpha": 0.1,
        #     "population_size": 1024,
        #     "return_mean_elites": False,  # trial.suggest_float("beta", 0.05, 0.5),
        #     "clipped_normal": False,
        #     "population_size_module": None,
        #     "population_decay_factor": 1.0,
        #     "colored_noise_exponent": 1.0,
        #     "initial_var_factor": 1.0,
        #     "lower_bound": ["${action_cfg.lower_bound}" for i in range(25)],
        #     "upper_bound": ["${action_cfg.upper_bound}" for i in range(25)],
        #     "keep_elite_frac": 1.0,
        #     "elite_shifting_n_step": 1,
        #     "provide_zero_action": True,
        #     "device": device,
        #     "batch_size": args_cli.num_envs,
        # },
        "robot_cfg": {
            "_target_": "fdm.sampling_planner.RobotCfg",
        },
        "to": {
            "_target_": "fdm.sampling_planner.SimpleSE2TrajectoryOptimizer",
            "action_cfg": "${action_cfg}",
            "robot_cfg": "${robot_cfg}",
            "to_cfg": "${to_cfg}",
            "optim": "${optim}",
            "device": device,
        },
    }

    # build planner
    planner = FDMPlanner(cfg, cfg_dict, args_cli=args_cli)
    # post modify runner and env
    planner = env_modifier_post_init(planner, args_cli=args_cli)

    # navigate
    planner.navigate()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
