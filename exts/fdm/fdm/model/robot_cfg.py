# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .fdm_model_cfg import FDMBaseModelCfg

###
# AOW
###


def tytan_model(cfg: FDMBaseModelCfg) -> FDMBaseModelCfg:
    """AOW model configuration."""

    # reduced number of observation, adjust the model sizes
    cfg.empirical_normalization_dim = 72
    cfg.state_obs_proprioception_encoder.input_size = 82

    return cfg


###
# AOW
###


def aow_model(cfg: FDMBaseModelCfg, env: str) -> FDMBaseModelCfg:
    """AOW model configuration."""

    # reduced number of observation, adjust the model sizes
    cfg.empirical_normalization_dim = 92
    cfg.state_obs_proprioception_encoder.input_size = 102

    if env == "height":
        cfg.obs_exteroceptive_encoder.out_channels = [32, 64, 128, 512]
        cfg.obs_exteroceptive_encoder.flatten = False
        cfg.obs_exteroceptive_encoder.avg_pool = True
    return cfg


###
# ANYmal Perceptive
###


def anymal_perceptive_model(cfg: FDMBaseModelCfg) -> FDMBaseModelCfg:
    """ANYmal Perceptive model configuration."""

    # extra cpg state, adjust the model sizes
    cfg.empirical_normalization_dim += 8
    cfg.state_obs_proprioception_encoder.input_size += 8

    return cfg
