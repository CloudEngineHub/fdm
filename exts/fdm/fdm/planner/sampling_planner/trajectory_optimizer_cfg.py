# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass


@dataclass
class ActionCfg:
    action_dim: int  # e.g. action_dim = 3 -> (x_vel, y_vel, yaw_vel)
    traj_dim: int  # Trajectory/Path length
    lower_bound: list[float]  # Units: forward - m/s, lateral - m/s, rad/s
    upper_bound: list[float]  # Units: forward - m/s, lateral - m/s, rad/s


@dataclass
class CEMCfg:
    num_iterations: int = 3
    elite_ratio: float = 0.03
    population_size: int = 128
    alpha: float = 0.1
    return_mean_elites: bool = False
    # CEM specific
    clipped_normal: bool = False  # ONLY CEM
    # ICEM specific
    population_size_module: int | None = None  # ONLY ICEM
    population_decay_factor: float = 1.0
    colored_noise_exponent: float = 1.0
    colored_noise_exponent_inital: float = 1.0
    initial_var_factor: float = 1.0

    keep_elite_frac: float = 1.0
    elite_shifting_n_step: int = 1  # TODO fix we have this variable twice now
    provide_zero_action: bool = True


@dataclass
class MPPICfg:
    num_iterations: int = 3
    population_size: int = 512
    gamma: float = 1.0
    """
    Reward scaling term.

    The gamma basically is a weighting based on the reward of each trajectory which trajectory should be focused the
    most when composing the next best "mean action" over all sampled trajectories. Same weight for all trajectories
    is achieved bei gamma = 0.The higher the number the more MPPI focuses on the high reward trajectories when
    computing the mean actions.
    """
    sigma: float = 0.95
    """
    Noise scaling term used in action sampling.

    It sets the upper bound for the variance when sampling actions - So e.g. if you now increase the lower and upper
    range for the sampling range of the yaw you should also increase strongly the sigma to allow for sampling more
    noisy trajectories.
    """
    beta: float = 0.3
    """
    Temporal time correlation

    So basically beta of 0 means the action is repeated for all timesteps within the same trajectory.A setting of 1
    means it is fully uncorrelated.
    """


@dataclass
class TrajectoryOptimizerCfg:
    dt: float = 0.1
    n_step_fwd: bool = True
    control: str = "position_control"
    init_debug: bool = True

    # FDM model needs minibatches due to memory restrictions
    batch_size: int = 5

    # Can be modified via dynamic reconfigure
    replan_every_n: int = 1
    debug: bool = False
    set_actions_below_threshold_to_0: bool = False
    vel_limit_lin: float = 0.1
    vel_limit_ang: float = 0.1

    # compute state costs from a cost map build from the height scan
    states_cost_w_cost_map: bool = False
    states_cost_w_cost_map_height_diff_thres: float = 0.3

    state_cost_w_action_rot: float = 1.0
    state_cost_w_action_trans_forward: float = 1.0
    state_cost_w_action_trans_side: float = 1.0

    state_cost_w_fatal_trav: float = 6.5
    state_cost_w_fatal_unknown: float = 10.0

    state_cost_w_risky_trav: float = 6.5
    state_cost_w_risky_unknown: float = 10.0

    state_cost_w_cautious_trav: float = 6.5
    state_cost_w_cautious_unknown: float = 10.0

    state_cost_w_early_goal_reaching: float = 2.0
    state_cost_early_goal_distance_offset: float = 0.3
    state_cost_early_goal_heading_offset: float = 0.3

    state_cost_w_early_stopping: float = 1.0

    state_cost_velocity_tracking: float = 1.0
    state_cost_desired_velocity: float = 1.0  # m/s

    pos_error_3d: bool = False
    """Compute the position error in 3D space instead of 2D space"""
    terminal_cost_w_rot_error: float = 10.0  # scales something between 0 and 1 (1 == opposite side)
    terminal_cost_w_position_error: float = 20.0  # scales something between 0 and x meters distance
    terminal_cost_close_reward: float = 100.0
    terminal_cost_distance_offset: float = 0.3  # only then the rotation reward is given
    terminal_cost_use_threshold: bool = True

    collision_cost_traj_factor: float = 0.5  # generally penalize higher collision probabilities
    collision_cost_high_risk_factor: float = 100.0  # cost for states with a collision probability above the threshold
    collision_cost_safety_factor: float = 0.0  # safety factor deducted from the collision threshold
    num_neighbors: int = 3  # sum the cost the neighbors to eliminate wrong collision predictions

    # Preprocessing [pp]
    #
    # Ramp function to define the cost
    #
    #
    #                      -------   fatal_value
    #                      |
    #                      |
    #               .------          risky_value
    #             .
    #           .
    # ---------                       0
    #
    #      safe_th  risky_th fatal_th
    pp_safe_th: float = 0.1
    pp_risky_th: float = 0.6
    pp_fatal_th: float = 0.8
    pp_risky_value: float = 0.5
    pp_fatal_value: float = 1.0


@dataclass
class RobotCfg:
    # Point
    resolution: float = 0.04
    # fatal = [
    #     ((0.01, 0.01), (0.015, 0.015)),  # Point
    # ]
    # risky = [
    #     ((0.0, 0.0), (0.01, 0.01)),  # Point
    # ]
    # cautious = [
    #     ((0.0, 0.0), (0.01, 0.01)),  # Point
    # ]

    # Rectangle Definitions for ANYmal D / hand measured by Jonas
    fatal = [
        ((-0.43, -0.235), (0.43, 0.235)),  # BODY
        ((0.43, -0.265), (0.63, 0.265)),  # Top Drive Area
        ((0.63, -0.125), (0.65, 0.125)),  # Top Face
        ((-0.63, -0.265), (-0.43, 0.265)),  # Bottom Drive Area
        ((-0.65, -0.125), (-0.63, 0.125)),  # Bottom Face
    ]

    # fatal = [
    #     ((-0.50, -0.2), (0.50, 0.2)),  # BODY is a small square
    # ]
    # fatal = [
    #     ((-0.45, -0.2), (0.45, 0.2)),  # Small BODY
    # ]
    risky = [((-0.70, -0.58), (0.70, 0.58))]
    cautious = [((-0.80, -0.68), (0.80, 0.68))]
