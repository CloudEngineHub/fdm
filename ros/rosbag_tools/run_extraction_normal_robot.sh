#!/bin/bash

# Extract all important files from the provided bags

BAG_PATH="/media/pascal/T7 Shield/FDMData/2024-09-10"  # base path to all the bags


## MISSION 2024-09-10

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-21-16.bag -o "${BAG_PATH}"/export_2024-09-10-09-21-16

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-23-47.bag -o "${BAG_PATH}"/export_2024-09-10-09-23-47

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-26-35.bag -o "${BAG_PATH}"/export_2024-09-10-09-26-35

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-43-41.bag -o "${BAG_PATH}"/export_2024-09-10-09-43-41

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-48-04.bag -o "${BAG_PATH}"/export_2024-09-10-09-48-04

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-48-25.bag -o "${BAG_PATH}"/export_2024-09-10-09-48-25

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-57-21.bag -o "${BAG_PATH}"/export_2024-09-10-09-57-21

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/_2024-09-10-09-59-40.bag -o "${BAG_PATH}"/export_2024-09-10-09-59-40


# MISSING RSL

# -- stairs 0
# /bin/python3 ./rosbag_extractor.py -bf "/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-10-15/fdm_relevant/merged.bag" -o "/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-10-15/fdm_relevant/export" --unified_frame odom -t "/elevation_mapping/elevation_map_raw" "/anymal_low_level_controller/actuator_readings" "/state_estimator/anymal_state" "/state_estimator/odometry" "/twist_mux/twist"

# -- stairs 1
/bin/python3 ./rosbag_extractor.py -bf "/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-13-11_stairs/fdm_relevant/merged.bag" -o "/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-13-11_stairs/fdm_relevant/export" --unified_frame odom -t "/elevation_mapping/elevation_map_raw" "/anymal_low_level_controller/actuator_readings" "/state_estimator/anymal_state" "/state_estimator/odometry" "/twist_mux/twist"

# -- meeting room
/bin/python3 ./rosbag_extractor.py -et 85.0 -bf "/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-21-14_meeting_room/fdm_relevant/merged.bag" -o "/media/pascal/T7 Shield/FDMData/2024-11-05-RSL/2024-11-05-20-21-14_meeting_room/fdm_relevant/export" --unified_frame odom -t "/elevation_mapping/elevation_map_raw" "/anymal_low_level_controller/actuator_readings" "/state_estimator/anymal_state" "/state_estimator/odometry" "/twist_mux/twist"
