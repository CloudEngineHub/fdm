

from __future__ import annotations

import torch
from collections.abc import Sequence

from omni.isaac.lab.envs import ManagerBasedRLEnv

from fdm.model.fdm_model_cfg import FDMBaseModelCfg

from .base_agent import Agent
from .time_correlated_actions_cfg import TimeCorrelatedCommandTrajectoryAgentCfg


class TimeCorrelatedCommandTrajectoryAgent(Agent):
    r"""Command generator that generates a velocity command in SE(2) from three different modes:

    1) linear time-correlated command sampling (Eqn.1),
    2) normal time-correlated command sampling (Eqn.2),
    3) and constant command sampling (Eqn.3)

    in the defined proportion.

    """

    cfg: TimeCorrelatedCommandTrajectoryAgentCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: TimeCorrelatedCommandTrajectoryAgentCfg, model_cfg: FDMBaseModelCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, model_cfg=model_cfg, env=env)
        # limits
        self._limits_min = torch.Tensor(
            (self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.ang_vel_z[0])
        ).to(self.device)
        self._limits_max = torch.Tensor(
            (self.cfg.ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_y[1], self.cfg.ranges.ang_vel_z[1])
        ).to(self.device)
        # -- time correlation constant for normal-correlation
        self.max_sigma = 0.5 * torch.tensor(
            [
                self.cfg.ranges.lin_vel_x[1] - self.cfg.ranges.lin_vel_x[0],
                self.cfg.ranges.lin_vel_y[1] - self.cfg.ranges.lin_vel_y[0],
                self.cfg.ranges.ang_vel_z[1] - self.cfg.ranges.ang_vel_z[0],
            ],
            device=self.device,
        )
        self.max_sigma_scale = self.cfg.sigma_scale
        # -- environment split
        assert (
            self.cfg.linear_ratio + self.cfg.normal_ratio + self.cfg.constant_ratio + self.cfg.regular_increasing_ratio
            == 1.0
        )
        self._uniform_envs_range = (0, int(self.env.num_envs * self.cfg.linear_ratio))
        self._normal_envs_range = (
            int(self.env.num_envs * self.cfg.linear_ratio),
            int(self.env.num_envs * (self.cfg.linear_ratio + self.cfg.normal_ratio)),
        )
        self._constant_envs_range = (
            int(self.env.num_envs * (self.cfg.linear_ratio + self.cfg.normal_ratio)),
            int(self.env.num_envs * (self.cfg.linear_ratio + self.cfg.normal_ratio + self.cfg.constant_ratio)),
        )
        self._regular_vel_envs_range = (
            int(self.env.num_envs * (self.cfg.linear_ratio + self.cfg.normal_ratio + self.cfg.constant_ratio)),
            self.env.num_envs,
        )

        # reset
        self.reset()

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "TimeCorrelatedCommandTrajectoryAgent:\n"
        msg += f"\tCommand dimension: {tuple(self._plan.shape)}\n"  # TODO: check that this is correct
        msg += f"\tPlanning horizon: {self.cfg.horizon}\n"
        return msg

    """
    Operations
    """

    def plan(self, env_ids: torch.Tensor, obs: torch.Tensor | None = None, random_init: bool = True):
        """Update the velocity commands which are correlated to the prev. command."""
        # check if there are any envs to plan
        if env_ids.shape[0] == 0:
            return

        if random_init:
            # sample the first command randomly
            self._random_command_sampler(env_ids)
        else:
            # use the last command as init
            self._plan[env_ids, 0] = self._plan[env_ids, self.cfg.horizon - 1]

        # update linear velocity correlation
        lin_corr_ids = torch.logical_and(env_ids >= self._uniform_envs_range[0], env_ids < self._uniform_envs_range[1])
        if lin_corr_ids.shape[0] > 0 and torch.any(lin_corr_ids):
            self._linear_time_correlated_command_sampler(env_ids[lin_corr_ids])
        # update normal velocity correlation
        normal_corr_ids = torch.logical_and(env_ids >= self._normal_envs_range[0], env_ids < self._normal_envs_range[1])
        if normal_corr_ids.shape[0] > 0 and torch.any(normal_corr_ids):
            self._normal_time_correlated_command_sampler(env_ids[normal_corr_ids])
        # regular velocity increasing commands
        regular_vel_ids = torch.logical_and(
            env_ids >= self._regular_vel_envs_range[0], env_ids < self._regular_vel_envs_range[1]
        )
        if regular_vel_ids.shape[0] > 0 and torch.any(regular_vel_ids):
            self._regular_velocity_command_sampler(env_ids[regular_vel_ids])
        # update constant velocity correlation
        constant_ids = env_ids[(env_ids >= self._constant_envs_range[0]) & (env_ids <= self._constant_envs_range[1])]
        self._plan[constant_ids] = self._plan[constant_ids, 0].unsqueeze(1).repeat(1, self.cfg.horizon, 1)

    """
    Command resamplers
    """

    def _random_command_sampler(self, env_ids: Sequence[int], horizon_idx: int = 0):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self._plan[env_ids, horizon_idx, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self._plan[env_ids, horizon_idx, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self._plan[env_ids, horizon_idx, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    def _linear_time_correlated_command_sampler(self, env_ids: Sequence[int]):
        """
        Sample time correlated command where the time correlation is defined by beta.
        """
        r = torch.empty(len(env_ids), device=self.device)
        for horizon_idx in range(self.cfg.horizon - 1):
            # precompute commands from uniform distribution
            self._random_command_sampler(env_ids, horizon_idx=horizon_idx + 1)
            # get correlated command
            opposite_beta = r.uniform_(0, 1 - self.cfg.max_beta).unsqueeze(1)
            self._plan[env_ids, horizon_idx + 1] = (
                self._plan[env_ids, horizon_idx] * (1 - opposite_beta)
                + self._plan[env_ids, horizon_idx + 1] * opposite_beta
            )
            # clip commands every step to ensure they are in range
            # clipping at the end can lead to longer command periods at the border of the range
            self._plan[env_ids, horizon_idx + 1] = torch.clip(
                self._plan[env_ids, horizon_idx + 1], min=self._limits_min, max=self._limits_max
            )

    def _normal_time_correlated_command_sampler(self, env_ids: Sequence[int]):
        """
        Sample time correlated command based on the normal distribution.
        """
        # sample new command with previous command as mean
        # sample correlation factor
        r = torch.empty(len(env_ids), device=self.device)
        sigma = torch.vstack((
            r.uniform_(0, self.max_sigma_scale),
            r.uniform_(0, self.max_sigma_scale),
            r.uniform_(0, self.max_sigma_scale),
        )).T
        sigma = sigma * self.max_sigma
        for horizon_idx in range(self.cfg.horizon - 1):
            # velocity command
            self._plan[env_ids, horizon_idx + 1] = torch.normal(self._plan[env_ids, horizon_idx], sigma)
            # clip commands every step to ensure they are in range
            # clipping at the end can lead to longer command periods at the border of the range
            self._plan[env_ids, horizon_idx + 1] = torch.clip(
                self._plan[env_ids, horizon_idx + 1], min=self._limits_min, max=self._limits_max
            )

    def _regular_velocity_command_sampler(self, env_ids: Sequence[int]):
        """
        Sample regular velocity increasing commands.
        """
        # velocity command
        self._plan[env_ids, :, 0] = torch.arange(
            self.cfg.ranges.lin_vel_x[0],
            self.cfg.ranges.lin_vel_x[1],
            (self.cfg.ranges.lin_vel_x[1] - self.cfg.ranges.lin_vel_x[0]) / self.cfg.horizon,
            device=self.device,
        )[None, :].repeat(len(env_ids), 1)
        self._plan[env_ids, :, 1] = torch.arange(
            self.cfg.ranges.lin_vel_y[0],
            self.cfg.ranges.lin_vel_y[1],
            (self.cfg.ranges.lin_vel_y[1] - self.cfg.ranges.lin_vel_y[0]) / self.cfg.horizon,
            device=self.device,
        )[None, :].repeat(len(env_ids), 1)
        self._plan[env_ids, :, 2] = (
            0.0  # torch.arange(self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1], self.cfg.horizon)[None, :].repeat(len(env_ids), 1)
        )
