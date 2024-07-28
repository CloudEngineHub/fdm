

import math
import torch

from .robot_shape import get_robot_shape
from .trajectory_optimizer_cfg import ActionCfg, RobotCfg, TrajectoryOptimizerCfg
from .utils import cosine_distance, get_se2, get_x_y_yaw


class SimpleSE2TrajectoryOptimizer:
    def __init__(
        self,
        action_cfg: ActionCfg,
        robot_cfg: RobotCfg,
        optim,
        to_cfg: TrajectoryOptimizerCfg,
        device: torch.device,
    ):
        """
        Initializes the SimpleSE2TrajectoryOptimizer with the given configurations and device.

        Args:
            action_cfg (ActionCfg): Configuration for the action space.
            robot_cfg (RobotCfg): Configuration for the robot footprint.
            to_cfg (TrajectoryOptimizerCfg): Configuration for the trajectory optimizer.
            optim (TODO): Underlying Black box optimizer
            device (torch.device): The device (CPU/GPU) on which to perform the computations.
        """

        self.action_cfg = action_cfg
        self.to_cfg = to_cfg
        self.device = device
        self.frame_id = "odom"
        self.fatal_xy, self.risky_xy, self.cautious_xy = get_robot_shape(robot_cfg, device)

        # Initialize Optimizer
        self.optim = optim

        # Buffer for previous solution
        self.previous_solution = None
        self.var = None

        # Set objective function
        if self.to_cfg.n_step_fwd:
            self.func = self.b_obj_func_N_step
        else:
            self.func = self.b_obj_func

        self.debug_info = {}

    ###
    # Operations
    ###

    def plan(self, obs: dict):
        """
        Initializes the observation dictionary with default values for planning.

        Args:
            obs (dict): A dictionary containing the following key-value pairs:
                - "goal": (torch.tensor, shape:=(BS, 3)): representing the goal with (x,y,yaw) in the odom frame.
                - "resample_population": bool: If the population should be resampled
                - "start": (torch.tensor, shape:=(BS, 3)): representing the start with (x,y,yaw) in the odom frame.
        Returns:
            torch.Tensor: The planned trajectory with shape (BS, TRAJ_LENGTH, STATE_DIM).
            torch.Tensor: The planned velocity with shape (BS, TRAJ_LENGTH, CONTROL_DIM).

        """

        self.obs = obs
        BS = self.obs["start"].shape[0]

        # MPPI
        population = None

        # Reset - Only needed for MPPI
        if self.previous_solution is None or self.obs["resample_population"]:
            self.optim.reset()

            # TODO here one can also interpolate the heading
            # TODO increase variance if goal is further away for sampling

            start_state = self.get_start_state(1, 1).clone()
            se2_odom_base = get_se2(start_state)  # Transformation from base to odom frame
            se2_odom_goal = get_se2(self.obs["goal"])  # Transformation from goal to odom frame
            se2_base_goal = torch.inverse(se2_odom_base) @ se2_odom_goal  # Transformation from base to goal frame
            xyyaw_base_goal = get_x_y_yaw(se2_base_goal)

            # Move omnidirectional and ignore heading.
            if True:
                dx = min(
                    math.ceil(
                        torch.abs(xyyaw_base_goal[0, 0, 0]) / (self.to_cfg.state_cost_desired_velocity * self.to_cfg.dt)
                    ),
                    25,
                )
                dy = min(
                    math.ceil(
                        torch.abs(xyyaw_base_goal[0, 0, 1]) / (self.to_cfg.state_cost_desired_velocity * self.to_cfg.dt)
                    ),
                    25,
                )
                x = torch.zeros((BS, 25, 1), device=self.device)
                y = torch.zeros((BS, 25, 1), device=self.device)

                x[:, :dx] = ((xyyaw_base_goal[:, 0, 0]) / dx)[:, None, None].repeat(1, dx, 1)
                y[:, :dy] = ((xyyaw_base_goal[:, 0, 1]) / dy)[:, None, None].repeat(1, dy, 1)
                z = torch.zeros((BS, 25, 1), device=self.device)

                population = torch.cat([x, y, z], dim=2) / self.to_cfg.dt
            else:
                # Currently the MPPI is not ideal for this given the time correlated sampling (maybe)
                norm_u = torch.sqrt(xyyaw_base_goal[0, 0, 0] ** 2 + xyyaw_base_goal[0, 0, 1] ** 2)
                cos_yaw = xyyaw_base_goal[0, 0, 0] / norm_u
                yaw_to_goal = torch.acos(cos_yaw)

                dyaw = min(
                    math.ceil(torch.abs(yaw_to_goal) / (1.5 * self.to_cfg.dt)),
                    25,
                )
                dx = min(
                    math.ceil(
                        torch.norm(xyyaw_base_goal[0, 0, :2])
                        / (self.to_cfg.state_cost_desired_velocity * self.to_cfg.dt)
                    ),
                    25 - dyaw,
                )

                x = torch.zeros((1, 25, 1), device=self.device)
                y = torch.zeros((1, 25, 1), device=self.device)
                yaw = torch.zeros((1, 25, 1), device=self.device)

                # Turn on spot
                yaw[:, :dyaw] = yaw_to_goal / dyaw
                x[:, dyaw : dyaw + dx] = torch.norm(xyyaw_base_goal[0, 0, :2]) / dx
                # y is fully 0
                population = torch.cat([x, y, yaw], dim=2) / self.to_cfg.dt

        best_population, self.var = self.optim.optimize(
            obj_fun=self.func,
            x0=population,
            var0=None,
            callback=self.logging_callback,
            batch_size=BS,
        )
        # self.var is shape := (BS, TRAJ_LENGTH, CONTROL_DIM)
        # best_population is shape := (BS, TRAJ_LENGTH, CONTROL_DIM)

        self.previous_solution = best_population
        states = self.func(best_population[None], only_rollout=True)
        # states is shape := (BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM)

        return states, best_population

    ###
    # FDM functions
    ###

    def b_obj_func_N_step(self, population: torch.Tensor, only_rollout: bool = False) -> torch.Tensor:
        """
        Objective function called by optimizer.
        We dynamicially allocate everything given that the population can grow or shrink
        """
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]
        TRAJ_LENGTH = population.shape[2]

        start_state = self.get_start_state(BS, NR_TRAJ).clone()

        if self.to_cfg.control == "velocity_control":
            # FIXME: add FDM model here
            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

            # If the actions is small make the robot stand
            if self.to_cfg.set_actions_below_threshold_to_0:
                m_vel_lin = torch.norm(population[:, :, :, :2], p=2, dim=3) < self.to_cfg.vel_limit_lin
                m_vel_ang = torch.abs(population[:, :, :, 2]) < self.to_cfg.vel_limit_ang
                m_vel_lin = m_vel_lin[:, :, :, None].repeat(1, 1, 1, 3)
                m_vel_lin[:, :, :, 2] = False
                m_vel_ang = m_vel_ang[:, :, :, None].repeat(1, 1, 1, 3)
                m_vel_ang[:, :, :, :2] = False

                if m_vel_lin.sum() > 0:
                    population[m_vel_lin] = 0
                if m_vel_ang.sum() > 0:
                    population[m_vel_ang] = 0

            # Integrate the velocity actions to positions
            actions = population * self.to_cfg.dt

            # Cumsum is an inplace operation therefore the clone is necesasry
            cummulative_yaw = actions.clone()[:, :, :, -1].cumsum(2)

            # We need to take the non-linearity by the rotation into account
            r_vec1 = torch.stack([torch.cos(cummulative_yaw), -torch.sin(cummulative_yaw)], dim=3)
            r_vec2 = torch.stack([torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=3)

            so2 = torch.stack([r_vec1, r_vec2], dim=4)

            # Move the rotation in time and fill first timestep with identity - see math chapter
            so2 = torch.roll(so2, shifts=1, dims=2)
            so2[:, :, 0, :, :] = torch.eye(2, device=so2.device)[None, None].repeat(BS, NR_TRAJ, 1, 1)

            # TODO this may be not needed given that max velocity in the x,y should stay enforced
            # actions[:,:,:,:2] /= torch.norm(actions[:,:,:,:2],p=2, dim=3)[None] * 1.2

            actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ actions[:, :, :, :2].contiguous().reshape(
                -1, 2, 1
            )
            actions_local_frame = actions_local_frame.contiguous().reshape(BS, NR_TRAJ, TRAJ_LENGTH, 2)
            cumulative_position = (actions_local_frame).cumsum(dim=2)
            states = torch.cat([cumulative_position, cummulative_yaw[:, :, :, None]], dim=3)

            # Transform the states from the current base frame to the odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(states)  # this here should be from base to points -> se2_points_base

            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

        elif self.to_cfg.control == "position_control":
            raise ValueError(
                "Not correctly implemented in the cost function handling the yaw actions forward, sidward motion"
                " correctly."
            )
            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM
            actions = population * self.to_cfg.dt
            states = start_state[:, :, None, :] + actions.cumsum(dim=2)

        if only_rollout:
            return states

        running_cost = self.states_cost(states.clone(), actions)
        if self.to_cfg.debug:
            self.debug_info["states_running_cost"] = running_cost.clone()

        running_cost = running_cost.mean(dim=2)
        terminal_cost = self.terminal_cost(states[:, :, -1])

        total_cost = running_cost + terminal_cost
        if self.to_cfg.debug:

            self.debug_info["terminal_cost"] = terminal_cost.clone()
            self.debug_callback(states, total_cost)

        # Return objective that needs to be maximized
        return -total_cost.T  # N_traj, BS

    def b_obj_func(self, population: torch.Tensor, only_rollout: bool = False) -> torch.Tensor:
        """
        Objective function called by optimizer.
        """
        # We dynamicially allocate all these things given that the population can grow or shrink
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]
        TRAJ_LENGTH = population.shape[2]

        state = self.get_start_state(BS, NR_TRAJ)
        STATE_DIM = state.shape[-1]

        if not only_rollout:
            running_cost = torch.zeros((BS, NR_TRAJ), device=self.device)

        if self.to_cfg or only_rollout:
            # state sequence
            states = torch.zeros((BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM), device=self.device)

        action = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM
        # actions = population * self.to_cfg.dt  # FIXME: this does currently not have a meaning

        # with Timer(f"full rollout time {self.debug}, {only_rollout}"):
        for i in range(TRAJ_LENGTH):
            if not only_rollout:
                running_cost += self.states_cost(state[:, :, None], action[:, :, i, :][:, :, None])[:, :, 0]

            if self.to_cfg or only_rollout:
                states[:, :, i, :] = state.clone()

            state = self.forward_dynamics(state, action[:, :, i, :])

        if self.to_cfg.debug:
            self.debug_callback(states, running_cost)

        if only_rollout:
            return states

        return -((running_cost + self.terminal_cost(state)).T)

    ###
    # Cost functions
    ###

    def states_cost(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluates state cost of a sequence of states.

        Args:
            states (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM)): Sequence of states
            actions (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, ACTION_DIM)): Sequence of actions

        Returns:
            (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH)): Sequence of costs per state
        """
        # Compute action cost
        control_effort_trans_forward = torch.abs(actions[:, :, :, 0]) * self.to_cfg.state_cost_w_action_trans_forward
        control_effort_trans_side = torch.abs(actions[:, :, :, 1]) * self.to_cfg.state_cost_w_action_trans_side
        control_effort_rot = torch.abs(actions[:, :, :, 2]) * self.to_cfg.state_cost_w_action_rot

        # Compute early_goal_bonus:
        position_offset = torch.norm(states[:, :, :, :2] - self.obs["goal"][:, None, None, :2], dim=3)
        # heading 0 if same; heading 1 if opposite
        goal_yaw = self.obs["goal"][:, None, None, 2].repeat(1, states.shape[1], states.shape[2])
        heading = cosine_distance(states[:, :, :, 2], goal_yaw) / 2

        m = (position_offset < self.to_cfg.state_cost_early_goal_distance_offset) * (
            heading < self.to_cfg.state_cost_early_goal_heading_offset
        )

        # Velocity tracking
        velocity_tracking_cost = (
            torch.abs(torch.norm(actions[:, :, :, :2], p=2, dim=3) - self.to_cfg.state_cost_desired_velocity)
            * self.to_cfg.state_cost_velocity_tracking
        )

        # No action cost if we reached the goal
        control_effort_trans_forward[m] = 0
        control_effort_trans_side[m] = 0
        control_effort_rot[m] = 0
        velocity_tracking_cost[m] = 0

        # Early goal reaching reward
        early_goal_cost = -torch.ones_like(control_effort_trans_forward) * self.to_cfg.state_cost_w_early_goal_reaching
        early_goal_cost[~m] = 0

        if self.to_cfg.debug:
            self.debug_info["states_control_effort_rot"] = control_effort_rot.clone()
            self.debug_info["states_control_effort_trans_forward"] = control_effort_trans_forward.clone()
            self.debug_info["states_control_effort_trans_side"] = control_effort_trans_side.clone()
            self.debug_info["states_early_goal_cost"] = early_goal_cost.clone()
            self.debug_info["velocity_tracking_cost"] = velocity_tracking_cost.clone()

        return (
            control_effort_trans_forward
            + control_effort_trans_side
            + control_effort_rot
            + early_goal_cost
            + velocity_tracking_cost
        )

    def terminal_cost(self, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the terminal state cost for a given state.

        Args:
            state (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, STATE_DIM)): The terminal state to evaluate.

        Returns:
            torch.Tensor: The calculated terminal cost for the given state, with shape (BS, NR_TRAJ).
        """

        position_offset = torch.norm(state[:, :, :2] - self.obs["goal"][:, None, :2], dim=2)

        # heading_cossine_distance is 0 if the same
        # heading_cossine_distance is 2 if opposite vectors
        heading_cossine_distance = cosine_distance(
            state[:, :, 2], self.obs["goal"][:, None, 2].repeat(1, state.shape[1])
        )
        # heading_reward -1 cost reduction if the same
        # heading_reward 0 cost reduction if opposite
        # heading_reward = heading_cossine_distance  # (-heading_cossine_distance / 2)-1
        # heading_reward[m] = 0

        if self.to_cfg.debug:
            self.debug_info["terminal_cost_position_offset"] = (
                position_offset.clone() * self.to_cfg.terminal_cost_w_position_error
            )
            self.debug_info["terminal_cost_heading_reward"] = (
                heading_cossine_distance.clone() * self.to_cfg.terminal_cost_w_rot_error
            )

        res = (
            position_offset * self.to_cfg.terminal_cost_w_position_error
            + heading_cossine_distance * self.to_cfg.terminal_cost_w_rot_error
        )

        if self.to_cfg.terminal_cost_use_threshold:
            m = position_offset < self.to_cfg.terminal_cost_distance_offset
            res[m] *= self.to_cfg.terminal_cost_close_reward

            if self.to_cfg.debug:
                self.debug_info["terminal_cost_total"] = res.clone()

        return res

    ###
    # Helper functions
    ###

    def get_start_state(self, batch_size: int, nr_traj: int) -> torch.Tensor:
        """
        Initializes the start state for a batch of trajectories.

        Args:
            batch_size (int): The batch size.
            nr_traj (int): The number of trajectories.

        Returns:
            torch.Tensor: The start state replicated for each trajectory, with shape (BS, NR_TRAJ, STATE_DIM).
        """
        return self.obs["start"].clone()[:, None, :].repeat(1, nr_traj, 1)

    def debug_callback(self, states, total_cost):
        b = 0
        best_traj = torch.argmin(total_cost[b])

        # Print statistics
        print("Running Cost:")
        for key in [
            "states_trav_cost",
            "states_control_effort_rot",
            "states_control_effort_trans_side",
            "states_control_effort_trans_forward",
            "states_early_goal_cost",
            "states_running_cost",
        ]:
            print(
                f"{key:>30}:    - Best Mean Total:"
                f" {round(self.debug_info[key][b, best_traj].mean().item(), 3):>10}       - Std Across Traj:"
                f" {round(self.debug_info[key][b].mean(dim=1).std().item(), 3):>10} "
            )
        #
        print("\nTerminal Cost:")
        for key in ["terminal_cost_position_offset", "terminal_cost_heading_reward", "terminal_cost"]:
            print(
                f"{key:>30}:    - Best Final: {round(self.debug_info[key][b, best_traj].item(), 3):>10}       - Std"
                f" Across Traj: {round(self.debug_info[key][b].std().item(), 3):>10} "
            )

    def logging_callback(self, population: torch.Tensor, values: torch.Tensor, iteration: int):
        if self.to_cfg.debug:
            min_v = values.min()
            print(f"Iteration: {iteration}, Values: {min_v}")

    def forward_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single sted forward dynamics model.
        """
        # Unit m + m/s * dt planning (actions are already integrated)
        return state + action
