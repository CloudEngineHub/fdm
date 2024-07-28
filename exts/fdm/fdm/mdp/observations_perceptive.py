

"""This sub-module contains the common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

from .actions import NavigationSE2Action
from .wild_anymal_obs import ProprioceptiveObservation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def wild_anymal(env: ManagerBasedEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Wild anymal observation term."""

    # extract the used quantities (to enable type-hinting)
    term: NavigationSE2Action = env.action_manager._terms[action_term]
    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "wild_anymal_obs"):
        env.wild_anymal_obs = ProprioceptiveObservation(
            num_envs=env.num_envs,
            device=env.device,
            simulation_dt=env.physics_dt,
            control_dt=env.physics_dt * term.cfg.low_level_decimation,
        )

    env.wild_anymal_obs.update(
        robot.data.joint_pos[:, asset_cfg.joint_ids],
        robot.data.joint_vel[:, asset_cfg.joint_ids],
        term.processed_actions,
        robot.data.root_lin_vel_b,
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
    )
    return env.wild_anymal_obs.get_obs(use_raisim_order=True)


def cgp_state(env: ManagerBasedEnv):
    """Return the phase of the CPG as a state."""

    if not hasattr(env, "wild_anymal_obs"):
        return torch.zeros(env.num_envs, 8, device=env.device)
    else:
        phase = env.wild_anymal_obs.cpg.get_phase()
        return torch.cat([torch.sin(phase).view(-1, 4, 1), torch.cos(phase).view(-1, 4, 1)], dim=2).view(-1, 8)
