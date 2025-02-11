cmake_minimum_required(VERSION 2.8.3)
project(fdm_navigation_ros)

find_package(catkin REQUIRED COMPONENTS
    dynamic_reconfigure
    rospy
    grid_map_msgs
    geometry_msgs
    nav_msgs
    series_elastic_actuator_msgs
    anymal_msgs
)

# catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
    dynamic_reconfigure/Base.cfg
)

catkin_package(
    CATKIN_DEPENDS rospy dynamic_reconfigure
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
    scripts/planner_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
