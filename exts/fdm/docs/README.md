# Forward Dynamics Model (FDM) - IsaacLab Extension

## Overview

The Forward Dynamics Model (FDM) is designed to predict the SE2 state of a robot given future actions. This extension integrates with IsaacLab to facilitate the training, testing, and fine-tuning of the dynamics model, as well as testing the planning capabilities.

## Code Structure

The codebase is organized into several directories, each serving a specific purpose:

### High-Level Structure

- **docs/**: Documentation files, including this README and the changelog.
- **fdm/**: Core implementation of the Forward Dynamics Model.
- **config/**: Configuration files for the extension.
  - `extension.toml`: Metadata and dependencies for the extension.
- **data/**: Contains data assets and related documentation.

### FDM Implementation

- **agents/**: Contains various agent implementations that generate commands for the robot. Each agent has a configuration file and a corresponding implementation file.
  - `base_agent.py`: Abstract base class for all agents.
  - `mixed_agent.py`: Combines multiple agents, each handling a subset of the environment.
  - `paper_figure_agent.py`: Generates commands for specific figures in publications.
  - `pink_noise_agent.py`: Uses pink noise for command generation.
  - `sampling_planner_agent.py`: Implements a sampling-based planner for command generation.
  - `time_correlated_actions.py`: Generates time-correlated velocity commands.

- **data_buffers/**: Manages data storage and retrieval for training and evaluation.
  - `dataset/`: Contains the `TrajectoryDataset` class for handling trajectory data.
  - `replay_buffer.py`: Implements a replay buffer for storing and sampling experiences.

- **env_cfg/**: Configuration for the environment setup, including UI elements for planner configuration.
  - `ui/planner_ui_window.py`: Provides a user interface for configuring planner settings, allowing users to adjust parameters and visualize planner behavior.

- **mdp/**: Includes Markov Decision Process (MDP) related configurations and noise models.
  - `noise/`: Contains noise configuration and testing scripts, such as `_test_perlin_noise.py`, which tests Perlin noise integration.
  - `curriculum/`: Implements curriculum learning strategies, like `command_ratio_curriculum_cfg.py`, to progressively increase task difficulty.

- **model/**: Contains the model architecture and related utilities for the FDM.
  - Defines the neural network structure used for predicting future states based on current observations and actions.

- **planner/**: Contains planning-related code, including trajectory optimization.
  - `planner.py`: Implements trajectory optimization algorithms to generate feasible paths for the robot, considering dynamic constraints.

- **runner/**: Manages the execution of training and evaluation loops.
  - Coordinates the training process, handles data loading, and manages the interaction between the model and the environment.

- **sensors/**: Handles sensor data processing and integration.
  - Processes raw sensor inputs to provide meaningful data for the model and agents.

- **utils/**: Utility functions and helpers used across the FDM implementation.
  - Provides common functions and classes that support various components of the FDM, such as mathematical operations and data normalization.

## Installation and Usage

Installation and usage instructions are given in the [README](../../../README.md).


## License

This project is licensed under the BSD-3-Clause License. See the LICENSE file for more details.

## Contact

For questions or feedback, please contact the maintainer, Pascal Roth, at [rothpa@ethz.ch].
