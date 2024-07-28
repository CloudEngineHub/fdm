

import torch

from .base_agent import Agent
from .pink_noise_agent_cfg import PinkNoiseAgentCfg
from .utils import powerlaw_psd_gaussian


class PinkNoiseAgent(Agent):
    cfg: PinkNoiseAgentCfg
    """Pink noise agent configuration."""

    def __init__(self, env, cfg: PinkNoiseAgentCfg):
        super().__init__(env, cfg)
        # reset
        self.reset(obs=None)

    def plan(self, obs: torch.Tensor | None = None, env_ids: torch.Tensor | None = None):
        # allow to reset individual envs
        if env_ids is None:
            env_ids = self._ALL_INDICES

        plan = powerlaw_psd_gaussian(
            self.cfg.colored_noise_exponent,
            size=(self.env.num_envs, self.action_dim, self.cfg.horizon),
            device=self.device,
        )

        plan = torch.minimum(plan * torch.sqrt(self.cfg.variance) + self.cfg.mean, self.cfg.upper_bound)
        plan = torch.maximum(plan, self.cfg.lower_bound)

        self._plan_step[env_ids] = 0
        self._plan[env_ids] = plan
        return plan
