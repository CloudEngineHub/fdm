#!/usr/bin/env bash

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

# Full Observation
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_VelAccLoss0.5
# Reduced Observation
./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion --reduced_obs --occlusions
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_UniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoTorque --reduced_obs --occlusions --remove_torque
# Noise + Reduced Observation
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NewHeightScanNoise --noise --reduced_obs --occlusions
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait7_Decay5e5_100k --noise --reduced_obs --occlusions --remove_torque
# Reduced Observation
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_reducedObs_friction_NoTorque --reduced_obs --friction --remove_torque

### Different robots
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_tytan --robot tytan --reduced_obs --remove_torque
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_tytan_quite --robot tytan_quiet --reduced_obs --remove_torque
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_SchedEp10_Wait7_Decay5e5_ReducedObs_noTorque_aow --robot aow --reduced_obs --remove_torque


### Continue training

# Full Observation
# ./docker/cluster/cluster_interface.sh job base fdm --run_name MergeSingleObjTerrain_HeightScan_lr3e3_Ep15_CR10_AllOnceStructure_NoPosLossCollSamples_ModPreTrained_lrData_Bs2048_HeightClipDoorOccCorr_resume  --resume "Aug08_12-49-55_PlannerRandomizedWallTerrain_HeightScan_lr3e3_AllOnceStructure_EngLoss_ModPreTrained_lrData_Bs2024_WideVelRange_noBN_WeightDecay"

# Timestamp 0.5 with 6sec horizon --> 12 prediction steps
# ${ISAACLAB_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_timeStamp0.5 --mode train --noise --reduced_obs --timestamp 0.5
# Timestamp 0.4 with 6sec horizon --> 15 prediction steps
# ${ISAACLAB_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_timeStamp0.4 --mode train --noise --reduced_obs --timestamp 0.4
# Timestamp 0.2 with 6sec horizon --> 30 prediction steps
# ${ISAACLAB_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_timeStamp0.2 --mode train --noise --reduced_obs --timestamp 0.2


### Real-World Training
# -- needs to happen with the reduced set of observations

# dilution 1
# ./docker/cluster/cluster_interface.sh job base fdm --mode train-real-world --run_name MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut_Noise_reducedObs_Occlusion_RealWorld_Dilution1 --noise --reduced_obs --occlusions --resume "Oct09_09-35-33_MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut_Noise_reducedObs_Occlusion" --real-world-dilution 1

# dilution 2
# ./docker/cluster/cluster_interface.sh job base fdm --mode train-real-world --run_name MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut_Noise_reducedObs_Occlusion_RealWorld_Dilution2 --noise --reduced_obs --occlusions --resume "Oct09_09-35-33_MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut_Noise_reducedObs_Occlusion" --real-world-dilution 2

# dilution 5
# ./docker/cluster/cluster_interface.sh job base fdm --mode train-real-world --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5_RealWorld_Dilution5_WithForest --noise --reduced_obs --occlusions --remove_torque --resume "Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5" --real-world-dilution 5

# dilution 10
# ./docker/cluster/cluster_interface.sh job base fdm --mode train-real-world --run_name MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5_RealWorld_Dilution10_WithForest --noise --reduced_obs --occlusions --remove_torque --resume "Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5" --real-world-dilution 10


### BASELINE
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --env baseline --run_name Baseline_NewEnv_NewCollisionShape_CorrLidar_UnifiedCollLoss_2DEnvPillar_NoBatchNorm
# ./docker/cluster/cluster_interface.sh job base fdm --mode train --env baseline --run_name Baseline_NewEnv_NewCollisionShape_CorrLidar_UnifiedCollLoss_2DEnvPillar_NoBatchNorm_noise --noise
