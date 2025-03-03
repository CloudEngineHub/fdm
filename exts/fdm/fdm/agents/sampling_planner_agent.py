# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import hydra
import omegaconf

from fdm.planner import get_planner_cfg

from .base_agent import Agent

# can only be imported if gui activated
try:
    from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
except ImportError:
    omni_debug_draw = None

if TYPE_CHECKING:
    from fdm.planner import SimpleSE2TrajectoryOptimizer
    from fdm.runner import FDMRunner

    from .sampling_planner_agent_cfg import SamplingPlannerAgentCfg


class SamplingPlannerAgent(Agent):
    """Command generator that uses a sampling-based planner to generate commands."""

    cfg: SamplingPlannerAgentCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: SamplingPlannerAgentCfg, runner: FDMRunner):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, runner=runner)

        # setup planner
        assert (
            self.cfg.horizon == self._runner.cfg.model_cfg.prediction_horizon
        ), "The horizon of the planner must match the prediction horizon of the model."
        planner_cfg = get_planner_cfg(
            self._runner.env.num_envs,
            self._runner.cfg.model_cfg.prediction_horizon,
            device=self._runner.env.device,
            population_size=self.cfg.population_size,
        )
        initialized_planner_cfg = omegaconf.OmegaConf.create(planner_cfg)
        self.planner: SimpleSE2TrajectoryOptimizer = hydra.utils.instantiate(initialized_planner_cfg.to)
        self.planner.set_fdm_classes(fdm_model=self._runner.model, env=self._runner.env)

        # limits
        self._limits_min = torch.Tensor(
            (self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.ang_vel_z[0])
        ).to(self.device)
        self._limits_max = torch.Tensor(
            (self.cfg.ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_y[1], self.cfg.ranges.ang_vel_z[1])
        ).to(self.device)

        # init debug draw
        if omni_debug_draw:
            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        else:
            self.draw_interface = None

    """
    Operations
    """

    def plan_reset(self, obs: dict, env_ids: torch.Tensor):
        # resample the population for environments that need to be replanned
        obs["planner_obs"]["resample_population"] = torch.zeros(
            self._runner.env.num_envs, dtype=torch.bool, device=self.device
        )
        obs["planner_obs"]["resample_population"][env_ids] = True

        # add states, proprio and extero observations to the planner obs
        obs["planner_obs"]["states"] = self._runner.replay_buffer.local_state_history[env_ids].clone()
        obs["planner_obs"]["proprio_obs"] = self._runner.replay_buffer.local_proprioceptive_observation_history[
            env_ids
        ].clone()
        obs["planner_obs"]["extero_obs"] = obs["fdm_obs_exteroceptive"][env_ids].clone()

        # replan for the environments
        with torch.inference_mode():
            curr_states, self._plan[env_ids] = self.planner.plan(
                obs=obs["planner_obs"], env_ids=env_ids, return_states=True
            )
        self._states[env_ids] = curr_states.squeeze(1)

    def plan(self, env_ids: torch.Tensor, obs: dict, random_init: bool = False):
        """Update the velocity commands which are correlated to the prev. command."""
        # check if there are any envs to plan
        if env_ids.shape[0] == 0:
            return

        # NOTE: if the environment previously collided then we will execute a random action instead of planning as the
        #       environment has not been reset immediately after the collision and the required observations are not
        #       updated yet. If an environment has collided before can be identified by the `random_init` flag.
        if random_init:
            # environment has collided in the step before, take a random action in the interval (-1, 1)
            # only first action of these will be executed, the rest will be overwritten in the process
            self._plan[env_ids] = (
                torch.rand((env_ids.shape[0], self.cfg.horizon, self.action_dim), device=self.device) * 2 - 1
            )
            # clip the actions to the limits
            self._plan[env_ids] = torch.clip(self._plan[env_ids], min=self._limits_min, max=self._limits_max)
        else:
            # environment did not collide but ran out of the planning horizon, replan directly
            self.plan_reset(obs=obs, env_ids=env_ids)

    def debug_viz(self, env_ids: list[int] | None = None):
        """Visualize the planning results."""
        # check if draw interface is available
        if self.draw_interface is None:
            return

        # check if there are any envs to visualize
        if env_ids is None:
            env_ids = self._ALL_INDICES.tolist()
        elif len(env_ids) == 0:
            return

        # clear the visualization
        self.draw_interface.clear_lines()

        for env_id in env_ids:
            state = self._states[env_id]
            state[:, 2] = 1.0

            self.draw_interface.draw_lines(
                state[:-1].tolist(),
                state[1:].tolist(),
                [(0, 0, 1, 1)] * int(state.shape[0] - 1),
                [5] * int(state.shape[0] - 1),
            )

    """
    Helper functions
    """

    def _init_buffers(self):
        """Initialize the buffers."""
        super()._init_buffers()

        # init buffer for the predicted states
        self._states = torch.zeros((self._runner.env.num_envs, self.cfg.horizon, 3), device=self.device)
