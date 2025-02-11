#!/bin/bash

# Extract from a merged bag with GPS odometry

# BAG_PATH="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-07-52-45_moenchsjoch_fenced_1"
# BAG_PATH="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-07-57-34_moenchsjoch_fenced_2"
# BAG_PATH="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-08-17-23_moenchsjoch_outside_1"
# BAG_PATH="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-03-08-42-30_moenchsjoch_outside_2"
# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/merged.bag -o "${BAG_PATH}"/export --unified_frame enu_origin \
#     -t "/elevation_mapping/elevation_map_raw" "/anymal_low_level_controller/actuator_readings" "/state_estimator/anymal_state" "/gt_box/inertial_explorer/tc/odometry" "/twist_mux/twist"


# Extract from merged bags with dlio SLAM

BAG_PATH="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-15-12-06-03_forest_albisguetli_slippery_slope"
# BAG_PATH="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-14-14-36-02_forest_kaeferberg_entanglement"
/bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/merged.bag -o "${BAG_PATH}"/export --unified_frame dlio_map \
    -t "/elevation_mapping/elevation_map_raw" "/anymal_low_level_controller/actuator_readings" "/state_estimator/anymal_state" "/dlio/lidar_map_odometry" "/twist_mux/twist"

# Extract all important files from the provided bags

# BAG_PATH="/media/pascal/T7 Shield/FDMData/2024-08-14-10-45-39"  # base path to all the bags
# BAG_FILE_PATTERN="2024-08-14-10-45-39"  # pattern to match the bag files

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/${BAG_FILE_PATTERN}_lpc_general_0.bag -o "${BAG_PATH}"/export -t /twist_mux/twist /anymal_low_level_controller/actuator_readings

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/${BAG_FILE_PATTERN}_lpc_state_estimator_0.bag -o "${BAG_PATH}"/export -t /state_estimator/anymal_state

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/${BAG_FILE_PATTERN}_nuc_cpt7_0.bag -o "${BAG_PATH}"/export -t /gt_box/cpt7/odom

# /bin/python3 ./rosbag_extractor.py -bf "${BAG_PATH}"/${BAG_FILE_PATTERN}_npc_elevation_mapping_0.bag -o "${BAG_PATH}"/export -t /elevation_mapping/elevation_map_raw
