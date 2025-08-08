# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class TrainerBaseCfg:
    """Configuration for the trainer."""

    # general training
    epochs: int = 8
    """Number of epochs for training with the collected samples of a single collection round."""
    early_stopping: bool = False
    """Whether to use early stopping."""
    lr_scheduler: bool = True
    """Whether to use a learning rate scheduler."""
    lr_scheduler_patience: int = 5  # Dec03 Model: 10;  Nov19 Model: 5
    """Patience for the learning rate scheduler."""
    learning_rate_warmup: int = 2  # Dec03 Model: 7;  Nov19 Model: 2
    """Number of collection rounds for learning rate warmup. Scheduling will be applied after warmup."""

    # data collection and dataloader
    num_samples: int = 80000  # Dec03 Model: 100000;  Nov19 Model: 80000
    """Number of trajectories to collect per collection round."""
    num_workers: int = 4
    """Number of workers for the dataloader."""
    shuffle_batch: bool = True
    """Whether to shuffle the batch during training."""
    batch_size: int = 2048
    """Batch size for training."""
    collision_rate: float | None = None  # 0.4
    """Rate of samples that are in collision."""
    # test_datasets: str | list[str] | None = None
    test_datasets: str | list[str] | None = [
        "{LOG_DIR}/test_datasets/plane.pkl",
        "{LOG_DIR}/test_datasets/PILLAR_EVAL_CFG.pkl",
        "{LOG_DIR}/test_datasets/STAIRS_WALL_EVAL_CFG.pkl",
        "{LOG_DIR}/test_datasets/STAIRS_RAMP_EVAL_CFG.pkl",
    ]
    """Static Test Datasets collected from different environments"""
    real_world_train_datasets: str | list[str] | None = None
    """Real world datasets for training."""
    real_world_test_datasets: str | list[str] | None = None
    """Real world datasets for testing."""
    small_motion_ratio: float | None = 0.1
    """Ratio of samples with small motion."""
    height_threshold: float | None = None
    """Filter samples based on height difference along the trajectory.

    .. note::
        This is a special parameter for testing set generation and is not used during training.
    """
    small_motion_threshold: float = 1.0
    """Threshold in meter to be considered a small motion.

    .. note::
        Only applied when the small_motion_ratio is set.
    """
    sample_filter_first_steps_coll: int = 0
    """Number of first steps to consider when removing samples that are in collision within these steps."""

    # noise models
    extereoceptive_noise_model: object | None = None
    """Noise model for the exteroceptive observations."""

    # weight decay and learning rate
    weight_decay: float = 1e-4  # Dec03 Model: 5e-5;  Nov19 Model: 1e-4
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
    wb_entity: str | None = None
    """Wandb entity name. If None, will use WANDB_ENTITY environment variable."""
    wb_mode: str | None = None
    """Wandb mode. If None, will use WANDB_MODE environment variable."""

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
    load_checkpoint: str = "model.*.pt"
    """The checkpoint file to load. Default is "model.*.pt" (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    apply_noise: bool = False
    """Whether to apply noise to the observations.

    .. note::
        The noise terms are defined in the environment configuration and passed to the trainer.
    """

    # ablation studies
    ablation_no_state_obs: bool = False
    """Whether to remove the state observations."""
    ablation_no_proprio_obs: bool = False
    """Whether to remove the proprioceptive observations."""
    ablation_no_height_scan: bool = False
    """Whether to remove the height scan."""
