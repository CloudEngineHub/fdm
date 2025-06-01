#!/usr/bin/env bash

ISAACLAB_HOME=/home/pascalr/orbit/IsaacLab

###
# Platform Test
###

# # ANYMAL
${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test.py --paper-platform-figure --robot "anymal_perceptive" \
    --record \
    --runs "Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque"

# AOW
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test.py --paper-platform-figure --robot "aow" \
#     --record \
#     --runs "Jan12_18-18-28_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_aow"

# TYTAN
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test.py --paper-platform-figure --robot "tytan" \
#     --record \
#     --runs "Jan12_18-13-10_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_tytan"

# TYTAN QUIET
# ${ISAACLAB_HOME}/isaaclab.sh -p ${ISAACLAB_HOME}/fdm_sub/scripts/test.py --paper-platform-figure --robot "tytan_quiet" \
#     --record \
#     --runs "Jan12_18-15-48_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_tytan_quite"
