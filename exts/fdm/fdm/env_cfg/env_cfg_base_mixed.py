# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import fdm.mdp as mdp

from .env_cfg_base import TERRAIN_ANALYSIS_CFG, FDMCfg, ObservationsCfg

##
# Ratio Split
##

PLANNER_RATIO = 0.0
RANDOM_RATIO = 1 - PLANNER_RATIO


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    command: mdp.MixedCommandCfg = mdp.MixedCommandCfg(
        terms={
            "null": mdp.MixedCommandCfg.SubsetCommandTermCfg(
                command_term=mdp.NullCommandCfg(),
                ratio=RANDOM_RATIO,
            ),
            "planner": mdp.MixedCommandCfg.SubsetCommandTermCfg(
                mdp.ConsecutiveGoalCommandCfg(
                    resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
                    debug_vis=False,
                    terrain_analysis=TERRAIN_ANALYSIS_CFG,
                ),
                ratio=PLANNER_RATIO,
            ),
        },
        resampling_time_range=(1000000.0, 1000000.0),
    )


@configclass
class PlannerObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PlannerObsCfg(ObsGroup):
        """Observations for the sampling based planner"""

        goal = ObsTerm(func=mdp.goal_command_w_se2_mixed, params={"command_name": "command", "subterm_name": "planner"})
        start = ObsTerm(func=mdp.se2_root_position)

        def __post_init__(self):
            self.concatenate_terms = False

    # Planner observations
    planner_obs: PlannerObsCfg = PlannerObsCfg()


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    command_ratios = CurrTerm(
        func=mdp.RandomPlannerCommandRatioCurriculumCfg.class_type(
            cfg=mdp.RandomPlannerCommandRatioCurriculumCfg(
                command_term_name="command",
                start_ratio=PLANNER_RATIO,
                end_ratio=0.4,
                update_interval=60000,
                update_step=0.1,
            )
        )
    )


##
# Environment configuration
##


@configclass
class MixedFDMCfg(FDMCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    observations: PlannerObservationsCfg = PlannerObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # NOTE: NOT NEEDED FOR CONSECUTIVE GOAL COMMANDS

        # self.events.reset_base = EventTerm(
        #     func=mdp.MixedRootResetEvent(
        #         cfg=mdp.MixedRootResetEventCfg(
        #             terms={
        #                 "analysis": mdp.MixedRootResetEventCfg.SubsetEventTermCfg(
        #                     event_term_cfg=EventTerm(
        #                         func=mdp.TerrainAnalysisRootReset(
        #                             cfg=TERRAIN_ANALYSIS_CFG,
        #                             robot_dim=0.5,
        #                         ),
        #                     ),
        #                     ratio=RANDOM_RATIO,
        #                 ),
        #                 "planner": mdp.MixedRootResetEventCfg.SubsetEventTermCfg(
        #                     event_term_cfg=EventTerm(func=mdp.reset_robot_position, mode="reset"),
        #                     ratio=PLANNER_RATIO,
        #                 ),
        #             },
        #         ),
        #     ),
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "yaw_range": (-3.14, 3.14),
        #         "velocity_range": {
        #             "x": (-0.5, 0.5),
        #             "y": (-0.5, 0.5),
        #             "z": (0, 0),
        #             "roll": (0, 0),
        #             "pitch": (0, 0),
        #             "yaw": (-0.5, 0.5),
        #         },
        #     },
        # )
