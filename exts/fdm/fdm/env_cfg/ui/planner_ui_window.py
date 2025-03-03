# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim
from isaaclab.envs.ui import ManagerBasedRLEnvWindow

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PlannerEnvWindow(ManagerBasedRLEnvWindow):
    """Window manager for the RL environment.

    On top of the basic environment window, this class adds controls for the RL environment.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "Orbit"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "Orbit".
        """
        import omni.ui  # noqa: F401
        from isaacsim.gui.components.ui_utils import dropdown_builder
        from omni.kit.window.extensions import SimpleCheckBox

        # initialize base window
        super().__init__(env, window_name)

        # cost visualization modes
        self.cost_viz_modes = [
            "None",
            "Cost",
            "Pose Reward",
            "Goal Distance",
            "Collision",
            "Goal Distance X",
            "Goal Distance Y",
            "Height Scan Cost",
        ]
        default_idx = 0
        self.current_cost_viz_mode = self.cost_viz_modes[default_idx]
        # perfect velocity visualization
        self.perfect_velocity = False

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    with omni.ui.HStack():
                        # add toggle between different visualization modes
                        self.ui_window_elements["planner_viz_mode"] = dropdown_builder(
                            "Planner Population Visualization",
                            items=self.cost_viz_modes,
                            default_val=default_idx,
                            on_clicked_fn=self.cost_viz_mode_changed,
                            tooltip=f"Select cost visualization mode. (default: {self.cost_viz_modes[default_idx]})",
                        )
                        isaacsim.gui.components.ui_utils.add_line_rect_flourish()
                        # add option to display perfect velocity estimate
                    with omni.ui.HStack():
                        text = "Display perfect velocity estimate."
                        omni.ui.Label(
                            "Perfect Velocity Rollout",
                            width=isaacsim.gui.components.ui_utils.LABEL_WIDTH - 12,
                            alignment=omni.ui.Alignment.LEFT_CENTER,
                            tooltip=text,
                        )
                        self.ui_window_elements["perfect_velocity"] = SimpleCheckBox(
                            model=omni.ui.SimpleBoolModel(),
                            checked=self.perfect_velocity,
                            on_checked_fn=self.perfect_velocity_changed,
                        )

                        isaacsim.gui.components.ui_utils.add_line_rect_flourish()

    def cost_viz_mode_changed(self, value: str):
        """Callback for cost visualization mode change."""
        self.current_cost_viz_mode = value

    def perfect_velocity_changed(self, value: bool):
        """Callback for perfect velocity change."""
        self.perfect_velocity = value
