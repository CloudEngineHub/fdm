# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple


def get_planner_cfg(
    traj_dim: int,
    optim: str,
    vel_range_x: Tuple[float, float],
    vel_range_y: Tuple[float, float],
    vel_range_yaw: Tuple[float, float],
    debug: bool,
    device: bool,
):
    """
    Get the configuration for the trajectory planner.

    Args:
        traj_dim: int: Trajectory/Path length
        optim: str: Optimizer to use (MPPI or iCEM)
        vel_range_x: Tuple[float, float]: Linear velocity range
        vel_range_y: Tuple[float, float]: Lateral velocity range
        vel_range_yaw: Tuple[float, float]: Angular velocity range
        debug: bool: Debug flag
        device: str: Device to run the optimizer on (cuda or cpu)
    """

    if optim == "MPPI":
        optim_cfg = {
            "_target_": "fdm_navigation.trajectory_optimizer.BatchedMPPIOptimizer",
            "num_iterations": 1,
            "population_size": 2048,
            "gamma": 2.0,  # 0.91 # Reward scaling term.
            "sigma": 0.87,  # 1.5,  #  Noise scaling term used in action sampling.
            "beta": 0.2,  # 0.08,  # Temporal time correlation
            "lower_bound": ["${action_cfg.lower_bound}" for i in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for i in range(traj_dim)],
            "provide_zero_action": True,
            "device": device,
        }

    elif optim == "iCEM":
        optim_cfg = {
            "_target_": "fdm_navigation.trajectory_optimizer.BatchedICEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.03,
            "alpha": 0.1,
            "population_size": 512,
            "return_mean_elites": False,  # trial.suggest_float("beta", 0.05, 0.5),
            "clipped_normal": False,
            "population_size_module": None,
            "population_decay_factor": 1.0,
            "colored_noise_exponent": 1.5,  # 1.0
            "initial_var_factor": 5.0,
            "lower_bound": ["${action_cfg.lower_bound}" for i in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for i in range(traj_dim)],
            "keep_elite_frac": 1.0,
            "elite_shifting_n_step": 1,
            "provide_zero_action": True,
            "device": device,
        }
    else:
        raise ValueError(f"Unknown Optimizer: {optim}")

    cfg_dict = {
        "action_cfg": {
            "_target_": "fdm_navigation.cfg.ActionCfg",
            "action_dim": 3,
            "traj_dim": traj_dim,
            "lower_bound": [vel_range_x[0], vel_range_y[0], vel_range_yaw[0]],
            "upper_bound": [vel_range_x[1], vel_range_y[1], vel_range_yaw[1]],
        },
        "to_cfg": {
            "_target_": "fdm_navigation.cfg.TrajectoryOptimizerCfg",
            "control": "fdm",  # "velocity_control",  #
            "init_debug": debug,
            "debug": debug,
            "dt": 0.5,
            "n_step_fwd": True,
            "set_actions_below_threshold_to_0": True,
            "vel_lin_min": 0.1,
            "vel_ang_min": 0.1,
            "state_cost_w_early_goal_reaching": 0.0,  # 200,  # 100
            "state_cost_w_early_stopping": 0.0,  # 30,  # 30
            "state_cost_w_action_trans_forward": 0,
            "state_cost_w_action_trans_side": 0,  # 60,
            "state_cost_w_action_rot": 0,  # 30,
            "state_cost_w_fatal_unknown": 0.0,  # 500,
            "state_cost_w_fatal_trav": 0.0,  # 1000,
            "state_cost_w_risky_unknown": 0,
            "state_cost_w_risky_trav": 0,
            "state_cost_w_cautious_unknown": 0,
            "state_cost_w_cautious_trav": 0,
            "state_cost_velocity_tracking": 0,
            "state_cost_desired_velocity": 0.0,  # 0.5,
            "state_cost_early_goal_distance_offset": 0.3,
            "state_cost_early_goal_heading_offset": 100,
            "terminal_cost_w_rot_error": 10,  # 20,
            "terminal_cost_w_position_error": 20,  # 60,
            "terminal_cost_distance_offset": 0.3,
            "terminal_cost_close_reward": 10,  # 500,
            "terminal_cost_use_threshold": True,  # no puling towards the goal
            "collision_cost_traj_factor": 0.0,
            "collision_cost_high_risk_factor": 100.0,  # 1000
            "pp_safe_th": 0.0,  # 0.3,
            "pp_risky_th": 0.0,  # 0.4,
            "pp_fatal_th": 0.0,  # 0.8,
            "pp_risky_value": 0.0,  # 0.5,
            "pp_fatal_value": 0.0,  # 1,
            "enable_timing": True,
        },
        "optim": optim_cfg,
        "robot_cfg": {"_target_": "fdm_navigation.cfg.RobotCfg"},
        "to": {
            "_target_": "fdm_navigation.trajectory_optimizer.SimpleSE2TrajectoryOptimizer",
            "action_cfg": "${action_cfg}",
            "robot_cfg": "${robot_cfg}",
            "to_cfg": "${to_cfg}",
            "optim": "${optim}",
            "device": device,
        },
    }
    return cfg_dict
