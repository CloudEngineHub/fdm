

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
    sigma: float = 0.95
    beta: float = 0.3


@dataclass
class TrajectoryOptimizerCfg:
    dt: float = 0.1
    n_step_fwd: bool = True
    control: str = "position_control"
    init_debug: bool = True

    # Can be modified via dynamic reconfigure
    replan_every_n: int = 1
    debug: bool = False
    set_actions_below_threshold_to_0: bool = True
    vel_limit_lin: float = 0.1
    vel_limit_ang: float = 0.1

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

    state_cost_velocity_tracking: float = 1.0
    state_cost_desired_velocity: float = 1.0  # m/s

    terminal_cost_w_rot_error: float = 10.0  # scales something between 0 and 1 (1 == opposite side)
    terminal_cost_w_position_error: float = 20.0  # scales something between 0 and x meters distance
    terminal_cost_close_reward: float = 100.0
    terminal_cost_distance_offset: float = 0.3  # only then the rotation reward is given
    terminal_cost_use_threshold: bool = True

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
