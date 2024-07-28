

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from .base_agent import AgentCfg


@configclass
class PinkNoiseAgentCfg(AgentCfg):
    upper_bound = 1.2
    """Upper bound for the actions."""
    lower_bound = -1.2
    """Lower bound for the actions."""
    colored_noise_exponent = 1.0
    """Exponent for the powerlaw PSD of the noise."""
    variance = 1.0
    """Variance of the noise."""
    mean = 0.0
    """Mean of the noise."""
