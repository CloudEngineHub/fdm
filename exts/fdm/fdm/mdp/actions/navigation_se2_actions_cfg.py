

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .navigation_se2_actions import NavigationSE2Action, PerceptiveNavigationSE2Action


@configclass
class NavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavigationSE2Action
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    low_level_policy_file: str = MISSING
    """Path to the low level policy file."""


@configclass
class PerceptiveNavigationSE2ActionCfg(NavigationSE2ActionCfg):
    class_type: type[ActionTerm] = PerceptiveNavigationSE2Action
    """ Class of the action term."""
    reorder_joint_list: list[str] = MISSING
    """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
    trained with a different order."""
