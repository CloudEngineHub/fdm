
from __future__ import annotations

import torch
from abc import ABC, abstractmethod

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.configclass import configclass

from fdm.model.fdm_model_cfg import FDMBaseModelCfg


@configclass
class AgentCfg:
    horizon: int = 200
    """Number of steps to plan ahead."""


class Agent(ABC):
    def __init__(self, cfg: AgentCfg, model_cfg: FDMBaseModelCfg, env: ManagerBasedRLEnv):
        self.env: ManagerBasedRLEnv = env
        self.cfg: AgentCfg = cfg
        self.model_cfg: FDMBaseModelCfg = model_cfg

        # init buffers
        self._init_buffers()

    """
    Properties
    """

    @property
    def device(self):
        return self.env.device

    @property
    def action_dim(self):
        return self.env.action_manager.action.shape[1]

    @property
    def resample_interval(self):
        return self.model_cfg.command_timestep / self.env.step_dt

    """
    Operations
    """

    def act(self, obs: torch.Tensor, dones: torch.Tensor, feet_contact: torch.Tensor):
        """Get the next actions for all environments.

        dones specifies the terminated environments for which the next command has to be resampled before the
        official resampling period.
        The function returns the next action for all environments. For all non-resampled environments, this will be
        the same as the previous action.
        """
        # for colliding environments, reset the action that is applied in the next step call and that will be recorded
        # for the current state
        colliding_envs = obs["fdm_state"][..., 7].to(torch.bool)
        if torch.any(colliding_envs):
            self.reset(obs=obs, env_ids=self._ALL_INDICES[colliding_envs], return_actions=False)

        # reset env counter when env is reset in simulation
        self.env_step_counter[dones] = 0

        # determine which environments should be updated depending on the sim time
        # NOTE: filter all environments on the first step
        # NOTE: filter all environments where not all feet have touched the ground yet
        updatable_envs = self.env_step_counter % self.resample_interval == 0
        updatable_envs[self.env_step_counter == 0] = False
        updatable_envs[~feet_contact] = False
        updatable_envs[self.env_step_counter == self.resample_interval] = ~colliding_envs[
            self.env_step_counter == self.resample_interval
        ]
        # for environments that should be resampled, increase the counter
        self._plan_step[updatable_envs] += 1

        # ensure to replan the environments out of a plan with their last step as new init
        env_to_replan = self._ALL_INDICES[self._plan_step >= (self.cfg.horizon - 1)]
        self.plan(env_ids=env_to_replan, obs=obs, random_init=False)
        self._plan_step[env_to_replan] = 0

        # in expectation of the next env step, increase the counter
        # NOTE: we only start the counting once all feet were in contact
        self.env_step_counter[feet_contact] += 1

        return self._plan[self._ALL_INDICES, self._plan_step]

    def reset(
        self, obs: torch.Tensor | None = None, env_ids: torch.Tensor | None = None, return_actions: bool = True
    ) -> torch.Tensor | None:
        # handle case for all envs
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset buffers
        self._plan_step[env_ids] = 0
        self._plan[env_ids] = 0
        # plan
        self.plan(env_ids=env_ids, obs=obs, random_init=True)
        if return_actions:
            return self._plan[self._ALL_INDICES, self._plan_step]

    @abstractmethod
    def plan(self, obs: torch.Tensor | None = None, env_ids: torch.Tensor | None = None, random_init: bool = True):
        pass

    """
    Helper functions
    """

    def _init_buffers(self):
        self._ALL_INDICES = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        # plan buffers
        self._plan_step = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self._plan = torch.zeros((self.env.num_envs, self.cfg.horizon, self.action_dim), device=self.device)
        # env step counter
        self.env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
