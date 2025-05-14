<h1 align="center">
  <br>
  Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation
  <br>
</h1>

<p align="center">

  [![Youtube Video](./docs/overview.png)](TODO)

</p>
<p align="center">
  <a href="https://leggedrobotics.github.io/fdm.github.io/">Project Page</a> •
  <a href="https://arxiv.org/abs/2504.19322">arXiv</a> •
  <a href="TODO">Video</a> •
  <a href="#citation">BibTeX</a>
</p>

<p align="center">

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

</p>

Our novel perceptive Forward Dynamics Model (FDM) enables real-time, learned traversability assessment for safe robot navigation by predicting future states based on environmental geometry and proprioceptive history. Trained in simulation and fine-tuned with real-world data, the model captures the full system dynamics beyond rigid body simulation. Integrated into a zero-shot Model Predictive Path Integral (MPPI) planner, our approach removes the need for tedious cost function tuning, improving safety and generalization. Tested on the ANYmal legged robot, our method significantly boosts navigation success in rough environments, with effective sim-to-real transfer.

⭐ If you find our perceptive FDM useful, star it on GitHub to get notified of new releases! The repository features:
- Implementation of the perceptive FDM training code as extension for [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- Integration of the perceptive FDM into a [Model Predictive Path Integral (MPPI)](https://ieeexplore.ieee.org/document/7989202) planner
- Real-world deployment of the perceptive FDM on the [ANYmal robot](https://rsl.ethz.ch/robots-media/anymal.html)


## Paper

A technical introduction to the theory behind our perceptive FDM is provided in our open-access RSS paper, available [here](https://arxiv.org/abs/2504.19322). For a quick overview, watch the accompanying 5-minute presentation [coming soon](TODO). More information about the work is available in the abstract below.

<details>
<summary>Abstract</summary>
<br>
Ensuring safe navigation in complex environments requires accurate real-time traversability assessment and understanding of environmental interactions relative to the robot's capabilities. Traditional methods, which assume simplified dynamics, often require designing and tuning cost functions to safely guide paths or actions toward the goal. This process is tedious, environment-dependent, and not generalizable. To overcome these issues, we propose a novel learned perceptive Forward Dynamics Model (FDM) that predicts the robot's future state conditioned on the surrounding geometry and history of proprioceptive measurements, proposing a more scalable, safer, and heuristic-free solution. The FDM is trained on multiple years of simulated navigation experience, including high-risk maneuvers, and real-world interactions to incorporate the full system dynamics beyond rigid body simulation. We integrate our perceptive FDM into a zero-shot Model Predictive Path Integral (MPPI) planning framework, leveraging the learned mapping between actions, future states, and failure probability. This allows for optimizing a simplified cost function, eliminating the need for extensive cost-tuning to ensure safety. On the legged robot ANYmal, the proposed perceptive FDM improves the position estimation by on average 41% over competitive baselines, which translates into a 27% higher navigation success rate in rough simulation environments. Moreover, we demonstrate effective sim-to-real transfer and showcase the benefit of training on synthetic and real data.
</details>

### Citation
```
@inproceedings{roth2025fdm,
  title={Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation},
  author={Roth, Pascal and Frey, Jonas and Cadena, Cesar and Hutter, Marco},
  booktitle={Robotics: Science and Systems (RSS 2025)},
  year={2025}
}
```

## Installation

### IsaacLab Extension (Training and Evaluation)

The extension is developed with [IsaacLab version 2.1.0](https://github.com/isaac-sim/IsaacLab/tree/v2.1.0) (latest tested commit 2e6946afb9b26f6949d4b1fd0a00e9f4ef733fcc). Future versions may work, but are not tested. IsaacLab runs on Ubuntu 20.04 - 24.04.

NOTE: Please use an IsaacLab version where [PR2393](https://github.com/isaac-sim/IsaacLab/pull/2393), [PR2394](https://github.com/isaac-sim/IsaacLab/pull/2394) and [PR2183](https://github.com/isaac-sim/IsaacLab/pull/2183) have been merged, they contain changes necessary to run the scripts successfully.

For details on the IsaacLab extensions, see the [README](exts/fdm/docs/README.md)

1. **Install IsaacSim and IsaacLab:**
   Follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to ensure IsaacSim and IsaacLab are installed.

2. **Add this repository as a submodule to your IsaacLab project:**
   ```bash
   cd IsaacLab
   git submodule add git@github.com:leggedrobotics/forward_dynamics_model.git fdm_sub
   ```
   **Important**: The submodule name cannot be `fdm` as this leads to import errors when using the pip installation from IsaacLab 4.2 upwards.

   Recursively update the submodule:
   ```bash
   git submodule update --init --recursive
   ```

   Download the assets from git lfs (Install instructions [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage))
   ```
   cd fdm_sub
   git lfs pull
   cd ..
   ```

3. **Link the extensions into the source directory:**
   ```bash
   cd source
   ln -s ../fdm_sub/exts/fdm .
   ```
   In addition, the FDM implementation depends on the [isaac-nav-suite](https://github.com/leggedrobotics/isaac-nav-suite) which is already included as a submodule in the FDM repo. Also the nav-suite extensions need to be linked into the source directory. Then leave the source directory.
   ```bash
   ln -s ../fdm_sub/isaac-nav-suite/exts/nav_suite .
   ln -s ../fdm_sub/isaac-nav-suite/exts/nav_tasks .
   cd ..
   ```

4. **Build the project:**
   All extensions are automatically build with the IsaacLab install functionality.

   ```bash
   ./isaaclab.sh -i
   ```

5. **Verify the installation:**
   To verify the installation, run the training script in debug mode:
   ```bash
   ./isaaclab.sh -p fdm_sub/scripts/train.py --mode debug
   ```
   The IsaacSim GUI should open and data collection should start, meaning the robots are moving. Later, the FDM training should start.

### ROS Integration (Real-World Deployment)

The integration is done with ROS Noetic on Ubuntu 20.04. For details on the ROS integration, see the [README](ros/README.md). It is recommended to use the docker image for a NVIDIA Jetson on the robot, which is provided [here](TODO).

1. **Create a catkin workspace:**
   ```bash
   mkdir -p fdm_ws/src
   cd fdm_ws/src
   ```

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/leggedrobotics/forward_dynamics_model.git
   ```

3. **Build the workspace:**
   ```bash
   cd ..
   catkin build fdm_navigation_ros
   ```

4. **Verify the installation:**

   Source the workspace and execute the planner.
   ```bash
   source devel/setup.bash
   roslaunch fdm_navigation_ros planner.launch
   ```

## Usage

### Model Demo

The latest model (`model.zip`) is available to download:
- [Simulation Model](https://drive.google.com/file/d/1_vkiW4cwW7f--ua7N7ZO5LR8-eOF3Zga/view?usp=sharing)
- [Real-World Fine-Tuned Model](https://drive.google.com/file/d/1mphOSa1ar3sw4IfVPSEcUnGXEN7wiBU9/view?usp=sharing)

**IMPORTANT** For evaluations in simulation, it is recommended to use the simulation model as it is not fitted for a particular real-world platform.

Follow the instructions above to setup the FDM extension for IsaacLab, then extract the model inside `IsaacLab/logs/fdm/fdm_se2_prediction_depth/fdm_latest`:

```bash
mkdir -p IsaacLab/logs/fdm/fdm_se2_prediction_depth/fdm_latest
cd IsaacLab/logs/fdm/fdm_se2_prediction_depth/fdm_latest
unzip model.zip
```

then run the test script for the dynamic estimation model:

```bash
./isaaclab.sh -p fdm_sub/scripts/test.py --runs fdm_latest
```

or for the planning:

```bash
./isaaclab.sh -p fdm_sub/scripts/plan_test.py --run fdm_latest --mode test
```

### Training

To train the Forward Dynamics Model, follow these steps:

1. **Simulation Pre-Training:**
   Run the training script:

   ```bash
   ./isaaclab.sh -p fdm_sub/scripts/train.py --mode train  --run_name <your-run-name>
   ```

   This will per default execute a training on ANYmal with a perceptive policy. For arguments that can be passed directly to the run script, see `./isaaclab.sh -p fdm_sub/scripts/train.py -h`. Detailed configuration for the environment, observation space, model, and much more is given in:

   - [IsaacLab Configs (mdp, terrain, robot, ...)](exts/fdm/fdm/env_cfg)
   - [Model Configs](exts/fdm/fdm/model)
   - [Trainer/ Runner Configs (optimizer, learning rate, collection intervals)](exts/fdm/fdm/runner)

   **Note**: For wandb logging during training, you need to set the following environment variables:
   - `WANDB_API_KEY`: Your Weights & Biases API key (required)
   - `WANDB_ENTITY`: Your Weights & Biases entity/username (can also be set in [TrainerCfg](exts/fdm/fdm/runner/trainer/trainer_cfg.py))
   - `WANDB_MODE`: Wandb mode, defaults to "online" if not set (can also be set in [TrainerCfg](exts/fdm/fdm/runner/trainer/trainer_cfg.py))

   During the training, evaluation steps in test environments can be performed. The data for those environments in typically pre-collected and remains constant for different trainings. The dataset paths can be defined in the [TrainerCfg](exts/fdm/fdm/runner/trainer/trainer_cfg.py). The collected of the datasets is done by executing:

   ```bash
   ./isaaclab.sh -p fdm_sub/scripts/test_data_collector.py --test_env <env-name>
   ```

2. **Real-World Data Collection:**

   You need to extract the data from the collected rosbags. The scripy relies on ROS Noetic, so it has to be executed with python 3.8. The script has been tested with the leggedrobot ANYmal (it likely does not work with other robots).

   ```bash
   python ros/rosbag_tools/rosbag_extractor.py \
      -bf <path_to_rosbag> \
      -o <path_to_output_dir> \
      --unified_frame odom \
      -t "/elevation_mapping/elevation_map_raw" "/anymal_low_level_controller/actuator_readings" "/state_estimator/anymal_state" "/state_estimator/odometry" "/twist_mux/twist"
   ```

   This extracts the data from the rosbag. To bring the data into the format required by the FDM, run the following script which at the same time will evaluate your current model on the collected data. Please adjust the configs inside the script to match the extracted data.

   ```bash
   ./isaaclab.sh -p scripts/real_world_test.py
   ```

   This process creates `train.pkl` and `val.pkl` files, used for fine tuning.
   The fine-tuning data used in this work is available [here](https://drive.google.com/file/d/1aFuScInnLjh2eukqOnnL-5ydqcUSpUbE/view?usp=sharing) and should be unzipped to `IsaacLab/logs/fdm/real_world_datasets`.

   The original rosbags are made available as part of the RSL GrandTour Dataset see [here](TODO).

3. **Real-World Fine-Tuning:**

   To fine-tune the model on the collected data, run the following script:

   ```bash
   ./isaaclab.sh -p fdm_sub/scripts/train.py --mode train-real-world  --run_name <your-run-name>  --real-world-dilution 10
   ```

   The real-world datasets are currently hard-coded in the `train.py` function, to the ones that can be downloaded as described above. Please adjust the paths for your own data in the `train.py` script.

### Evaluation

1. **Dynamics Estimation**

   - **Qualitative** evaluation

      A qualitative evaluation of the dynamics estimation in **defined test environments** can be executed using the following command. This executed as many commands as the prediction horizon is long and then stops the robot to then give time to closly evaluate the predictions.

      ```bash
      ./isaaclab.sh -p fdm_sub/scripts/test.py --runs <run-name>
      ```

      Given the argument `--paper-figure` generates Fig. **TODO** of the paper. Executing the script with `--paper-platform-figure` and varying the platforms using `--robot` argument generates Fig. **TODO**.

      **IMPORTANT**: This script exports the policies to `jit` format and therefore has to be executed before the execution of the ROS code.

      A qualitative evaluation in **any environment** can be executed using the following command. Here longer trajectories are executed and the predictions are shown continuously along them.

      ```bash
      ./isaaclab.sh -p fdm_sub/scripts/eval.py --runs <run-name> --terrain-cfg <terrain-name>
      ```

      Possible terrain names are defined in the terrain config in [TerrainConfigs](exts/fdm/fdm/env_cfg/terrain_cfg.py).

   - **Quantitative** evaluation

      A **quantitative** evaluation between the constant velocity assumption, the baseline method by Kim et al., and the developed method is executed with the following command:

      ```bash
      ./isaaclab.sh -p fdm_sub/scripts/eval_metrics.py --runs <run-name>
      ```

   - **Real-World** evaluation

      Evaluation on the real-world data is done by running:

      ```bash
      ./isaaclab.sh -p fdm_sub/scripts/real_world_eval_metrics.py --runs <run-name>
      ```

2. **Planning**

   The planning evaluation is done by calling:

   ```bash
   ./isaaclab.sh -p fdm_sub/scripts/plan_test.py --run <run-name>
   ```

   The script runs with different modes:
   - `--mode test`: **qualitative** evaluation in a test environment
   - `--mode metric --env_type 2D`: **qualitative** evaluation in a 2D environment
   - `--mode metric --env_type 3D`: **qualitative** evaluation in a 3D environment
   - `--mode plot`: generate the Fig. **TODO** of the paper

   The predicted paths and their rewards can be visualized using `--cost_show` argument.

   While this function uses pre-defined environments with start-goal pairs selected so that the planner has to overcome/ avoid certain obstacles, a evaluation with random start-goal pairs can be executed by running:

    ```bash
   ./isaaclab.sh -p fdm_sub/scripts/plan_metric.py --run <run-name>
   ```

   As this involves many straight paths without significant obstacles, the `plan_test.py` script provide a more representative evaluation.
   For planning without

### Real-World Deployment

Policies have to be `jit` exported before running the ROS code. The export is performed when running the `test.py` script as described above.
Given the export, adjust the path in the [config file](ros/fdm_navigation_ros/config/default.yaml).
Then execute the following steps to run the planner:

1. **Source the workspace:**
   ```bash
   source devel/setup.bash
   ```

2. **Run the planner:**
   ```bash
   roslaunch fdm_navigation_ros planner.launch
   ```

## Contributing

Contributions are welcome! Please ensure that your code adheres to the project's coding standards by running the defined pre-commit formatter.
Please note that due to different python version of ROS (python 3.8) and IsaacLab (> python 3.10), both have to be installed on the system.

```bash
pre-commit run --all-files
```


### License

This code belongs to Robotic Systems Lab, ETH Zurich.
All rights reserved.

**Authors: [Pascal Roth](https://pascal-roth.github.io/), [Jonas Frey](https://jonasfrey96.github.io/), [Cesar Cadena](https://scholar.google.ch/citations?hl=en&user=aOns5HQAAAAJ), and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<br />
Maintainer: Pascal Roth, rothpa@ethz.ch**

The FDM inference part (ROS directory) has been tested under ROS Noetic on Ubuntu 20.04.
The training code has been tested in Ubuntu 24.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.
