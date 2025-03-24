#!/usr/bin/env bash

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

###
# OURS
###

# # Collect data
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --headless --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --headless

# # Collect data with reduced observation
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --reduced_obs --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --reduced_obs --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --headless --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --reduced_obs --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --reduced_obs --headless

# # Collect data with reduced observation and removed torque
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --reduced_obs --remove_torque --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --reduced_obs --remove_torque --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --remove_torque --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --remove_torque --headless --terrain_analysis_points 2250
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --remove_torque --headless --height_threshold 0.1
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --remove_torque --headless --height_threshold 0.1 --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --reduced_obs --remove_torque --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --reduced_obs --remove_torque --headless

# # Collect data with reduced observation and removed torque and occlusions
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --reduced_obs --remove_torque --occlusions --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless --terrain_analysis_points 2250
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless --height_threshold 0.1
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless --height_threshold 0.1 --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --reduced_obs --remove_torque --occlusions --headless

# # Collect data with reduced observation and noise
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --reduced_obs --noise --occlusions --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --reduced_obs --noise --occlusions --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --noise --occlusions --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --noise --occlusions --headless --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --reduced_obs --noise --occlusions --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --reduced_obs --noise --occlusions --headless

# # Collect data with reduced observation and noise
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --reduced_obs --remove_torque --noise --occlusions --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --reduced_obs --remove_torque --noise --occlusions --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --reduced_obs --remove_torque --noise --occlusions --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --reduced_obs --remove_torque --noise --occlusions --headless --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --reduced_obs --remove_torque --noise --occlusions --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --reduced_obs --remove_torque --noise --occlusions --headless

###
# BASELINE
###

# collect data with baseline model
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --env baseline --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --env baseline --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --env baseline --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --env baseline --headless --terrain_analysis_points 2250
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --env baseline --headless --height_threshold 0.1
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --env baseline --headless --height_threshold 0.1 --terrain_analysis_points 2250
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --env baseline --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --env baseline --headless

# # Collect data with baseline model and noise
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "plane" --noise --env baseline --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "PILLAR_EVAL_CFG" --noise --env baseline --headless --terrain_analysis_points 7500
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_WALL_EVAL_CFG" --noise --env baseline --headless
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_RAMP_EVAL_CFG" --noise --env baseline --headless --terrain_analysis_points 2250
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "GRID_EVAL_CFG" --noise --env baseline --headless
# # ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test_data_collector.py --test_env "STAIRS_EVAL_CFG" --noise --env baseline --headless
