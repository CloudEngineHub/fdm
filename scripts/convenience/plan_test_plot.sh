#!/usr/bin/env bash

# Generate the videos and plots

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Cost" \
#     --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Goal_Distance"  \
#     --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Collision" \
#     --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Pose_Reward" \
#     --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Cost" \
#     --env heuristic \
#     --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Goal_Distance" \
#     --env heuristic \
#     --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Height_Scan_Cost" \
#     --env heuristic \
#     --reduced_obs --occlusion --remove_torque

${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
    --mode plot \
    --cost_show "Pose_Reward" \
    --env heuristic \
    --reduced_obs --occlusion --remove_torque
