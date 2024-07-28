

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from ..sensor_noise_models import DepthCameraNoiseCfg
from .fdm_exteroception_model import FDMExteroceptionModel
from .fdm_model_cfg import FDMDepthHeightScanModelCfg
from .model_base_cfg import BaseModelCfg


@configclass
class FDMExteroceptionModelCfg(FDMDepthHeightScanModelCfg):
    """Configuration class for the FDM exteroception model."""

    class_type: type[FDMExteroceptionModel] = FDMExteroceptionModel

    loss_weights = {"depth_reconstruction": 1.0, "height_map": 1.0}

    target_height_map_size: int = None
    """Filled out during configuration of the environment."""

    depth_camera_noise: DepthCameraNoiseCfg | None = None
    """Noise model for the depth camera.

    Here we include the noise model in the model configuration, as the ground truth depth map is required as target.
    In the normal setup, the noise model would be part of the runner configuration. If None, no noise is added.
    """

    def __post_init__(self):
        super().__post_init__()


@configclass
class FDMPreTrainedExteroceptionModelCfg(FDMDepthHeightScanModelCfg):
    """Configuration class for the FDM exteroception model."""

    class_type: type[FDMExteroceptionModel] = FDMExteroceptionModel

    loss_weights = {"depth_reconstruction": 1.0, "height_map": 1.0}

    target_height_map_size: int = None
    """Filled out during configuration of the environment."""

    depth_camera_noise: DepthCameraNoiseCfg | None = None
    """Noise model for the depth camera.

    Here we include the noise model in the model configuration, as the ground truth depth map is required as target.
    In the normal setup, the noise model would be part of the runner configuration. If None, no noise is added.
    """

    obs_exteroceptive_encoder: BaseModelCfg.PerceptNetCfg = BaseModelCfg.PerceptNetCfg(
        layers=[2, 2, 2, 2],
        avg_pool=True,
    )

    def __post_init__(self):
        super().__post_init__()
