#!/usr/bin/env bash

# RUN THE EVALUATION FOR PLANNER

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

###
# Define Models
###

MODEL_FDM="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque"
BASELINE_FDM_MODEL="Mar04_17-42-57_Baseline_UnifiedCollLoss_BatchNorm"   # "Mar04_13-22-10_Baseline_NewEnv_NewCollisionShape_CorrLidar_UnifiedCollLoss_2DEnvPillar_BatchNorm"

###
# 2D Environment
###

${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
    --run ${MODEL_FDM} \
    --mode metric \
    --env_type 2D \
    --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --run ${BASELINE_FDM_MODEL} --env baseline \
#     --mode metric \
#     --env_type 2D \

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --run ${MODEL_FDM} --env heuristic \
#     --mode metric \
#     --env_type 2D \
#     --reduced_obs --occlusion --remove_torque

###
# 3D Environment
###

${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
    --run ${MODEL_FDM} \
    --mode metric \
    --env_type 3D \
    --reduced_obs --occlusion --remove_torque

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --run ${BASELINE_FDM_MODEL} --env baseline \
#     --mode metric \
#     --env_type 3D \

# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/plan_test.py \
#     --run ${MODEL_FDM} --env heuristic \
#     --mode metric \
#     --env_type 3D \
#     --reduced_obs --occlusion --remove_torque
