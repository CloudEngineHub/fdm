#!/usr/bin/env bash

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

###
# Platform Test
###

# # ANYMAL
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm/scripts/test.py --paper-platform-figure --robot "anymal_perceptive" \
#     --reduced_obs --remove_torque --occlusions \
#     --runs "Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque"
#     # --runs "Dec03_20-25-59_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoTorque"

# AOW
${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm/scripts/test.py --paper-platform-figure --robot "aow" \
    --reduced_obs --remove_torque \
    --runs "Jan12_18-18-28_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_aow"

# TYTAN
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm/scripts/test.py --paper-platform-figure --robot "tytan" \
#     --reduced_obs --remove_torque \
#     --runs "Jan12_18-13-10_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_tytan"
#     # --runs "Oct02_18-29-13_MergeSingleObjTerrain_HeightScanSquareLarge_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_tytan"

# TYTAN QUIET
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm/scripts/test.py --paper-platform-figure --robot "tytan_quiet" \
#     --reduced_obs --remove_torque \
#     --runs "Jan12_18-15-48_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_tytan_quite"
#     # --runs "Oct02_18-31-09_MergeSingleObjTerrain_HeightScanSquareLarge_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_tytan_quiet"
