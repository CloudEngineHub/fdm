# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .env_cfg_base import TERRAIN_ANALYSIS_CFG, FDMCfg
from .env_cfg_base_mixed import PLANNER_RATIO, RANDOM_RATIO, MixedFDMCfg
from .env_cfg_baseline import FDMBaselineEnvCfg
from .env_cfg_depth import FDMDepthCfg, PreTrainingFDMDepthCfg
from .env_cfg_height import FDMHeightCfg, MixedFDMHeightCfg
from .env_cfg_heuristic_planner import FDMHeuristicsHeightCfg
from .terrain_cfg import (
    FDM_EVAL_EXTEROCEPTIVE_TERRAINS_CFG,
    FDM_EXTEROCEPTIVE_TERRAINS_CFG,
    FDM_ROUGH_TERRAINS_CFG,
    FDM_TERRAINS_CFG,
)
