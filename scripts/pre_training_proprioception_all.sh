#!/usr/bin/env bash

ORBIT_HOME=/home/pascal/orbit/orbit

# Full Observation
# ${ORBIT_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_fullObs --mode train --noise
# Reduced Observation
${ORBIT_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_friction --mode train --noise --reduced_obs --friction

# Timestamp 0.5 with 6sec horizon --> 12 prediction steps
# ${ORBIT_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_timeStamp0.5 --mode train --noise --reduced_obs --timestamp 0.5
# Timestamp 0.4 with 6sec horizon --> 15 prediction steps
# ${ORBIT_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_timeStamp0.4 --mode train --noise --reduced_obs --timestamp 0.4
# Timestamp 0.2 with 6sec horizon --> 30 prediction steps
# ${ORBIT_HOME}/docker/container.sh job proprio --run_name atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm_Noise_MeanVelAccLoss_reducedObs_timeStamp0.2 --mode train --noise --reduced_obs --timestamp 0.2
