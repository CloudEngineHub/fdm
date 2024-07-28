

from __future__ import annotations

import os

from omni.isaac.lab_assets import ISAACLAB_ASSETS_EXT_DIR

from omni.isaac.lab.utils import configclass


@configclass
class TrainerBaseCfg:
    """Configuration for the trainer."""

    # general training
    epochs: int = 15
    """Number of epochs for training with the collected samples of a single collection round."""
    clip_grad: bool = False
    """Whether to clip the gradient."""
    max_grad_norm: float = 2.0
    """Max gradient norm for clipping."""
    early_stopping: bool = False
    """Whether to use early stopping."""
    learning_rate_warmup: int = 2
    """Number of collection rounds for learning rate warmup. Scheduling will be applied after warmup."""

    # data collection and dataloader
    num_samples: int = 80000  # e.g. 50000 with 4096 envs, approx. 12 samples per env
    """Number of trajectories to collect per collection round."""
    num_workers: int = 4
    """Number of workers for the dataloader."""
    shuffle_batch: bool = True
    """Whether to shuffle the batch during training."""
    batch_size: int = 2048
    """Batch size for training."""
    collision_rate: float | None = None
    """Rate of samples that are in collision."""
    test_datasets: str | list[str] | None = [
        os.path.join(ISAACLAB_ASSETS_EXT_DIR, "Terrains", "navigation_terrain_stairs.pkl"),
        os.path.join(ISAACLAB_ASSETS_EXT_DIR, "Terrains", "navigation_terrain_perlin_stepping_stones.pkl"),
        os.path.join(ISAACLAB_ASSETS_EXT_DIR, "Terrains", "navigation_terrain_ramp_platform.pkl"),
    ]
    """Static Test Datasets collected from different environments"""

    # noise models
    extereoceptive_noise_model: object | None = None
    """Noise model for the exteroceptive observations."""

    # weight decay and learning rate
    weight_decay: float = 0
    """Weight decay for the optimizer."""
    learning_rate: float = 3e-3
    """Learning rate for the optimizer."""

    # saving and logging
    logging: bool = True
    """Whether to log the training."""
    experiment_name: str = "fdm_se2_prediction_depth"
    """Name of the experiment. """
    run_name: str | None = None
    """Name of the run."""
    wb_entity: str = "rothpa"
    wb_api_key: str = "8d9b2277691e6b27dc2861ce2bc7c0148113c3ce"
    wb_mode: str = "online"
    """Wandb project name and api key."""

    # resume and load checkpoint
    resume: bool = False
    """Whether to resume. Default is False."""
    encoder_resume: dict | None = None
    """Resume encoders and fix their weights.

    Should be passed as a dictionary with the encoder name as key and the path to the checkpoint as value.
    """
    encoder_resume_add_to_optimizer: bool = True
    """Whether to add the encoders to the optimizer after the first learning rate decrease."""
    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """
    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
