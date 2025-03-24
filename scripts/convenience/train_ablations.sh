#!/usr/bin/env bash

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

# Baseline model
./docker/cluster/cluster_interface.sh \
    job base fdm \
    --mode train \
    --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque_base \
    --reduced_obs \
    --occlusions \
    --remove_torque

# Ablation no state obs
./docker/cluster/cluster_interface.sh \
    job base fdm \
    --mode train \
    --ablation_mode no_state_obs \
    --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque_no_state_obs \
    --reduced_obs \
    --occlusions \
    --remove_torque

# Ablation no proprio obs
./docker/cluster/cluster_interface.sh \
    job base fdm \
    --mode train \
    --ablation_mode no_proprio_obs \
    --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque_no_proprio_obs \
    --reduced_obs \
    --occlusions \
    --remove_torque

# Ablation no height scan
./docker/cluster/cluster_interface.sh \
    job base fdm \
    --mode train \
    --ablation_mode no_height_scan \
    --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque_no_height_scan \
    --reduced_obs \
    --occlusions \
    --remove_torque
