#!/usr/bin/env bash

# start large scale planner eval
./docker/cluster/cluster_interface.sh job base planner --mode full --run Oct09_09-27-48_MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut
