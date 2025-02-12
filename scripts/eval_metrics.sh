#!/usr/bin/env bash

# Run the metric evaluation for a model, the baseline and different configurations.

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

# Evaluate without noise
# MODEL_FDM="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque"
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm/scripts/eval_metrics.py --runs $MODEL_FDM --reduced_obs --remove_torque --occlusions

# Evaluate with noise
NOISE_MODEL_FDM="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5"
${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm/scripts/eval_metrics.py --runs ${NOISE_MODEL_FDM} --reduced_obs --remove_torque --noise --occlusions
