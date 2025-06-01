#!/usr/bin/env bash

# Generate the videos and plots

ISAACLAB_HOME=/home/pascalr/orbit/IsaacLab

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Cost" \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Goal_Distance"  \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Collision" \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Pose_Reward" \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Cost" \
#     --env heuristic \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Goal_Distance" \
#     --env heuristic \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Height_Scan_Cost" \
#     --env heuristic \
#     --occlusion

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --mode plot \
#     --cost_show "Pose_Reward" \
#     --env heuristic \
#     --occlusion


## VIDEO

${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
    --mode plot_video \
    --cost_show "Cost" \
    --occlusion
