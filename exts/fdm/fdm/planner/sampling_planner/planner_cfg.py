# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner configuration"""

from fdm import VEL_RANGE_X, VEL_RANGE_Y, VEL_RANGE_YAW


def get_planner_cfg(
    num_envs: int, traj_dim: int = 10, debug: bool = False, device: str = "cuda", population_size: int = 1024
) -> dict:
    cfg_dict = {
        "traj_dim": traj_dim,
        "action_cfg": {
            "_target_": "fdm.planner.ActionCfg",
            "action_dim": 3,
            "traj_dim": traj_dim,
            "lower_bound": [VEL_RANGE_X[0], VEL_RANGE_Y[0], VEL_RANGE_YAW[0]],
            "upper_bound": [VEL_RANGE_X[1], VEL_RANGE_Y[1], VEL_RANGE_YAW[1]],
        },
        "to_cfg": {
            "_target_": "fdm.planner.TrajectoryOptimizerCfg",
            "init_debug": debug,
            "debug": debug,
            "dt": 0.5,  # 0.20,
            "n_step_fwd": True,
            "control": "fdm",
            "state_cost_w_early_goal_reaching": 0.0,  # 200,
            "state_cost_w_early_stopping": 0.0,  # 30,
            "state_cost_w_action_trans_forward": 0,
            "state_cost_w_action_trans_side": 0,  # 60,
            "state_cost_w_action_rot": 0,  # 30,
            "state_cost_w_fatal_unknown": 0,  # 500,
            "state_cost_w_fatal_trav": 0,  # 1000,
            "state_cost_w_risky_unknown": 0,
            "state_cost_w_risky_trav": 0,
            "state_cost_w_cautious_unknown": 0,
            "state_cost_w_cautious_trav": 0,
            "state_cost_velocity_tracking": 0,
            "state_cost_desired_velocity": 0.0,  # 1.0,
            "state_cost_early_goal_distance_offset": 0.3,
            "state_cost_early_goal_heading_offset": 100,
            "terminal_cost_w_rot_error": 0,  # 20,
            "terminal_cost_w_position_error": 10,  # 60,
            "terminal_cost_distance_offset": 0.3,
            "terminal_cost_close_reward": 10,  # 500,
            "terminal_cost_use_threshold": True,  # no puling towards the goal
            "collision_cost_traj_factor": 0.0,  # 0.5
            "collision_cost_high_risk_factor": 1000,  # 1000
            "pp_safe_th": 0.3,
            "pp_risky_th": 0.4,
            "pp_fatal_th": 0.8,
            "pp_risky_value": 0.5,
            "pp_fatal_value": 1,
            # for heurists over cost map
            "states_cost_w_cost_map": False,
            # mini batch size for FDM
            "batch_size": 15000,
        },
        "optim": {
            "_target_": "fdm.planner.BatchedMPPIOptimizer",
            "num_iterations": 1,
            "population_size": population_size,
            "gamma": 2.0,  # 0.91,  # FIXME: debug for baseline planning  (before 1.0)
            "sigma": 0.87,  # 0.87,
            "beta": 0.2,  # 0.08, 0.5  # FIXME: debug for baseline planning  (before 0.6)
            "lower_bound": ["${action_cfg.lower_bound}" for i in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for i in range(traj_dim)],
            "device": device,
            "batch_size": num_envs,
        },
        # "optim": {
        #     "_target_": "fdm.planner.BatchedICEMOptimizer",
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
        #     "batch_size": num_envs,
        # },
        "robot_cfg": {
            "_target_": "fdm.planner.RobotCfg",
        },
        "to": {
            "_target_": "fdm.planner.SimpleSE2TrajectoryOptimizer",
            "action_cfg": "${action_cfg}",
            "robot_cfg": "${robot_cfg}",
            "to_cfg": "${to_cfg}",
            "optim": "${optim}",
            "device": device,
        },
    }

    return cfg_dict
