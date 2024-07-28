

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import carb

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from fdm.mdp import GoalCommand

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def illegal_contact_delayed(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, delay: int = 1
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold after a certain delay.

    This allows to record the observations when the robot is in contact and do not directly terminate.
    Useful in navigation tasks/ forward dynamics model learning.
    The delay is multiplied by the decimation of the low-level policy to make sure that the obstacles
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # make sure that delay is not larger than the history length
    delay_physics_timestep = delay * env.cfg.decimation
    if delay_physics_timestep >= net_contact_forces.shape[1]:
        carb.log_warn(
            f"Delay {delay} requires a history length of {delay_physics_timestep} but current length is only"
            f" {net_contact_forces.shape[1]}.Setting delay to {net_contact_forces.shape[1] - 1}"
        )
        delay_physics_timestep = net_contact_forces.shape[1] - 1
    # check if any contact force exceeds the threshold
    # newest contact force is at history idx 0, so delay is removing the newest contact force
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, delay_physics_timestep:, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold,
        dim=1,
    )


###
# Planner
###


def at_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    speed_threshold: float = 0.25,
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        speed_threshold: The speed threshold at the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms["command"]
    # check for termination
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_geneator.pos_command_w[:, :2], dim=1, p=2)
    abs_velocity = torch.norm(asset.data.root_vel_w[:, 0:6], dim=1, p=2)
    # Check conditions
    within_goal = distance_goal < distance_threshold
    within_speed = abs_velocity < speed_threshold
    # Return termination
    return within_goal & within_speed
