# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Extract Real World Data from Rosbags in the format to build a TrajectoryDataset.
"""

import argparse
import contextlib
import copy
import numpy as np
import os
import time
import tqdm
from typing import List, Optional

import rosbag
import roslaunch
import rospy
import tf2_geometry_msgs
import tf2_ros
from anymal_msgs.msg import AnymalState
from geometry_msgs.msg import PointStamped, TwistStamped
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path
from rosgraph import rosenv
from sensor_msgs.msg import CompressedImage, Image
from series_elastic_actuator_msgs.msg import SeActuatorReadings


def check_roscore():
    # Check if roscore is already running
    try:
        master_uri = rosenv.get_master_uri()
    except rospy.service.ServiceException:
        master_uri = None

    # If roscore is not running, start a new instance
    if master_uri is None:
        print("roscore is not running, starting a new instance...")
        roscore_launch = roslaunch.scriptapi.ROSLaunch()
        roscore_launch.start()
        roscore_launch.launch(roslaunch.pmon.REQUIRED, roslaunch.rlutil.resolve_launch_arguments(["roscore"]))
        master_uri = rosenv.get_master_uri()

        # Wait for roscore to start up
        while master_uri is None:
            time.sleep(1)
            with contextlib.suppress(rospy.service.ServiceException):
                master_uri = rosenv.get_master_uri()

        print("roscore started successfully.")
    else:
        print("roscore is already running.")

    return master_uri


def setup_tf_buffer(bag: rosbag.Bag, tf_bags: Optional[List[str]] = None) -> tf2_ros.Buffer:
    """
    Gets the transform from frame A to frame B from a rosbag /tf message at a specific timestamp.

    Parameters:
        bag (rosbag.Bag): Rosbag with a /tf message.

    Returns:
        tf_buffer (tf2_ros.Buffer): A tf buffer with the transforms from the rosbag.
    """
    print("Setting up tf buffer...", end=" ")
    tf_buffer = tf2_ros.Buffer(rospy.Duration(bag.get_end_time() - bag.get_start_time()))

    # get addtional bags for the tf_buffer
    if tf_bags is None:
        bags = [bag]
    else:
        bags = [bag] + [rosbag.Bag(tf_bag, "r") for tf_bag in tf_bags]

    for curr_bag in bags:
        for topic, msg, t in curr_bag.read_messages(topics=["/tf", "/tf_static"]):
            for msg_tf in msg.transforms:
                if topic == "/tf_static":
                    tf_buffer.set_transform_static(msg_tf, "default_authority")
                else:
                    tf_buffer.set_transform(msg_tf, "default_authority")
    print("Done")
    return tf_buffer


def get_intrinsics(bag: rosbag.Bag, topics: List[str], image_topic: str) -> np.ndarray:
    """
    Get camera intrinsics from rosbag
    """
    topic_type = topics[image_topic].msg_type

    if topic_type == Image._type:
        cam_parent = os.path.split(image_topic)[0]
        camera_info_topic = f"{cam_parent}/camera_info"
        # Check if the CameraInfo topic is in the list of topics
        if camera_info_topic in topics:
            # Print a message indicating that the CameraInfo topic was found
            print(f"Corresponding CameraInfo topic found: {camera_info_topic}")
            for _, msg, _ in bag.read_messages(topics=[camera_info_topic]):
                K = np.array(msg.K).reshape(3, 3)
                height = msg.height
                width = msg.width
                break
            assert K is not None, f"Could not find camera info under topic {camera_info_topic}"
        else:
            # Print a message indicating that the CameraInfo topic was not found
            print(f"Corresponding CameraInfo topic not found for {image_topic}")
    elif topic_type == CompressedImage._type:
        cam_parent = image_topic.split("/")[1]

        # Check if the CameraInfo topic is in the list of topics
        if f"/{cam_parent}/camera_info" in topics:
            camera_info_topic = f"/{cam_parent}/camera_info"
            # Print a message indicating that the CameraInfo topic was found
            print(f"Corresponding CameraInfo topic found: {camera_info_topic}")
        elif f"/{cam_parent}/{image_topic.split('/')[2]}/camera_info" in topics:
            camera_info_topic = f"/{cam_parent}/{image_topic.split('/')[2]}/camera_info"
        else:
            # Print a message indicating that the CameraInfo topic was not found
            raise ValueError(f"Corresponding CameraInfo topic not found for {image_topic}")

        for _, msg, _ in bag.read_messages(topics=[camera_info_topic]):
            K = np.array(msg.K).reshape(3, 3)
            height = msg.height
            width = msg.width
            break
        assert K is not None, f"Could not find camera info under topic {camera_info_topic}"

    else:
        raise ValueError(f"Topic {image_topic} is of type {topic_type} which is not supported!")

    return K, height, width


def main(args):  # noqa: C901
    """
    Extract all necessary data for the FDM from the recorded rosbags.
    """
    bag = rosbag.Bag(args.bag_file, "r")
    print(f"Opened rosbag {args.bag_file} with {bag.get_message_count()} messages")
    bag_topics = bag.get_type_and_topic_info().topics

    if args.end_time is None:
        args.end_time = bag.get_end_time() - bag.get_start_time()

    # init cv_bridge
    # bridge = CvBridge()

    # check if the topics are inside the rosbag
    for used_topic in args.topics:
        assert used_topic in bag_topics, f"Topic {used_topic} not in bag!"

    # init buffers
    buffers = {}
    metadata = {}
    counters = {}
    buffer_path = {}

    for topic in args.topics:
        # init counter
        counters[topic] = 0
        # init path
        buffer_path[topic] = os.path.join(args.output_dir, topic[1:].replace("/", "_"))
        os.makedirs(buffer_path[topic], exist_ok=True)
        # init buffer and potentially save intrinsics for images
        if bag_topics[topic].msg_type == SeActuatorReadings._type:
            # joint position (12), t_sec, t_nsec
            buffers[topic] = np.zeros((bag.get_message_count(topic), 14))
            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write(
                    "Joint Positions\nLF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA,"
                    " RH_HFE, RH_KFE, t_sec, t_nsec"
                )

        elif bag_topics[topic].msg_type == TwistStamped._type:
            # lin_vel_x, lin_vel_y, ang_vel_z, t_sec, t_nsec
            buffers[topic] = np.zeros((bag.get_message_count(topic), 5))
            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write("lin_vel_x, lin_vel_y, ang_vel_z, t_sec, t_nsec")

        elif bag_topics[topic].msg_type == Image._type:
            buffers[topic] = np.zeros((bag.get_message_count(topic), 9))  # x, y, z, qx, qy, qz, qw, t_sec, t_nsec
            intrinsics, height, depth = get_intrinsics(bag, bag_topics, topic)
            np.savetxt(os.path.join(buffer_path[topic], "intrinsics.txt"), intrinsics)

        elif bag_topics[topic].msg_type == AnymalState._type:
            # joint position (12), joint velocity (12), joint torques (12), lin vel (3), ang vel (3), t_sec, t_nsec
            buffers[topic] = np.zeros((bag.get_message_count(topic), 51))
            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write(
                    "Joint Positions (12) Joint Velocities (12) Joint Torques (12) Linear Velocities (3) Angular"
                    " Velocities (3) Base pos (3) Base rot (4 - qx, qy, qz, qw) t_sec, t_nsec"
                )

        elif bag_topics[topic].msg_type == Odometry._type:
            # x, y, z, qx, qy, qz, qw, t_sec, t_nsec
            buffers[topic] = np.zeros((bag.get_message_count(topic), 9))
            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write("x, y, z, qx, qy, qz, qw, t_sec, t_nsec")

        elif bag_topics[topic].msg_type == GridMap._type:
            # dim_0, dim_1
            buffers[topic] = np.zeros((bag.get_message_count(topic), 200, 200))
            # additional buffer for the time and position of the grid map  --> x, y, z, qx, qy, qz, qw, len_x, len_y, resolution, t_sec, t_nsec
            metadata[topic] = np.zeros((bag.get_message_count(topic), 12))
            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write("Grid Map (200, 200)")
            with open(os.path.join(buffer_path[topic], "meta_format.txt"), "w") as f:
                f.write("x, y, z, qx, qy, qz, qw, len_x, len_y, resolution, t_sec, t_nsec")

        elif bag_topics[topic].msg_type == PointStamped._type:
            buffers[topic] = np.zeros((bag.get_message_count(topic), 5))
            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write("Goal in odom frame \nx, y, z, t_sec, t_nsec")

        elif bag_topics[topic].msg_type == Path._type:
            # x, y, z, t_sec, t_nsec
            buffers[topic] = np.zeros((bag.get_message_count(topic), 10, 3))  # FIXME: remove hardcoding of path length
            # current goal in odom frame when the path was published
            metadata[topic] = np.zeros((bag.get_message_count(topic), 5))

            # save data format
            with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                f.write("Path in odom frame \nx, y, z, t_sec, t_nsec")
            with open(os.path.join(buffer_path[topic], "meta_format.txt"), "w") as f:
                f.write("Goal in odom frame \nx, y, z, t_sec, t_nsec")

        else:
            raise ValueError(f"Topic {topic} is of type {bag_topics[topic].msg_type} which is not supported!")

    # get transform between odom and cameras
    check_roscore()  # roscore needs to run to use tf_buffer
    tf_buffer = setup_tf_buffer(bag, args.tf_bags)

    # configure process bar
    pbar = tqdm.tqdm(total=sum([bag.get_message_count(topic) for topic in args.topics]))

    init_time = None

    for topic, msg, t in bag.read_messages(topics=args.topics):
        if init_time is None:
            init_time = t.to_sec()

        if (t.to_sec() - init_time) < args.start_time or (t.to_sec() - init_time) > args.end_time:
            continue

        topic_type = msg._type

        if topic_type == AnymalState._type:
            # save joint order names in the first iteration
            if counters[topic] == 0:
                joint_names = msg.joints.name
                # write to file
                with open(os.path.join(buffer_path[topic], "data_format.txt"), "w") as f:
                    f.write("\n".join(joint_names))

            # extract joint information and current base velocities
            buffers[topic][counters[topic]] = np.array([
                msg.joints.position  # joint position
                + msg.joints.velocity  # joint velocity
                + msg.joints.effort  # joint torque
                + (
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                )  # linear velocity in base frame
                + (
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z,
                )  # angular velocity in base frame
                + (
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                )  # position in base frame
                + (
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                )  # orientation in base frame
                + (msg.header.stamp.secs,)
                + (msg.header.stamp.nsecs,)
            ])
            counters[topic] += 1

        elif topic_type == SeActuatorReadings._type:
            # extract low-level controller command
            # joints are in the following order: "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE", "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"
            buffers[topic][counters[topic]] = np.array(
                [single_reading.state.joint_position for single_reading in msg.readings]  # joint position
                + [msg.readings[0].header.stamp.secs, msg.readings[0].header.stamp.nsecs]  # time
            )
            counters[topic] += 1

        elif topic_type == TwistStamped._type:
            # extract high-level planner command
            buffers[topic][counters[topic]] = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.angular.z,
                msg.header.stamp.secs,
                msg.header.stamp.nsecs,
            ])
            counters[topic] += 1

        elif topic_type == Odometry._type:
            if topic == "/gt_box/inertial_explorer/odometry" or topic == "/gt_box/inertial_explorer/tc/odometry":
                # FIXME: remove once the topic of the box has been corrected
                try:
                    # transform = tf_buffer.lookup_transform("base", "box_base", msg.header.stamp)
                    transform = tf_buffer.lookup_transform("enu_origin", "base", msg.header.stamp)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logerr(f"Transform failed of msg {topic_type} at time {t.to_time()}. Disregard message")
                    continue

                buffers[topic][counters[topic]] = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                    msg.header.stamp.secs,
                    msg.header.stamp.nsecs,
                ])
                counters[topic] += 1

            elif topic == "/dlio/lidar_map_odometry":

                try:
                    # transform = tf_buffer.lookup_transform("base", "box_base", msg.header.stamp)
                    transform = tf_buffer.lookup_transform("dlio_map", "base", msg.header.stamp)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logerr(f"Transform failed of msg {topic_type} at time {t.to_time()}. Disregard message")
                    continue

                buffers[topic][counters[topic]] = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                    msg.header.stamp.secs,
                    msg.header.stamp.nsecs,
                ])
                counters[topic] += 1

            else:
                try:
                    # transform = tf_buffer.lookup_transform(msg.header.frame_id, msg.header.frame_id, msg.header.stamp)
                    transform = tf_buffer.lookup_transform(args.unified_frame, msg.header.frame_id, msg.header.stamp)
                    # transform = tf_buffer.lookup_transform(args.unified_frame, "box_base", msg.header.stamp)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logerr(f"Transform failed of msg {topic_type} at time {t.to_time()}. Disregard message")
                    continue

                # raise NotImplementedError("Odometry is not supported yet!")
                transformed_pose = tf2_geometry_msgs.do_transform_pose(msg.pose, transform)
                buffers[topic][counters[topic]] = np.array([
                    transformed_pose.pose.position.x,
                    transformed_pose.pose.position.y,
                    transformed_pose.pose.position.z,
                    transformed_pose.pose.orientation.x,
                    transformed_pose.pose.orientation.y,
                    transformed_pose.pose.orientation.z,
                    transformed_pose.pose.orientation.w,
                    msg.header.stamp.secs,
                    msg.header.stamp.nsecs,
                ])
                counters[topic] += 1

        elif topic_type == GridMap._type:

            try:
                # transform = tf_buffer.lookup_transform(msg.info.header.frame_id, msg.info.header.frame_id, msg.info.header.stamp)
                transform = tf_buffer.lookup_transform(
                    args.unified_frame, msg.info.header.frame_id, msg.info.header.stamp
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr(f"Transform failed of msg {topic_type} at time {t.to_time()}. Disregard message")
                continue

            transformed_pose = tf2_geometry_msgs.do_transform_pose(msg.info, transform)

            metadata[topic][counters[topic]] = np.array([
                transformed_pose.pose.position.x,
                transformed_pose.pose.position.y,
                transformed_pose.pose.position.z,
                transformed_pose.pose.orientation.x,
                transformed_pose.pose.orientation.y,
                transformed_pose.pose.orientation.z,
                transformed_pose.pose.orientation.w,
                msg.info.length_x,
                msg.info.length_y,
                msg.info.resolution,
                msg.info.header.stamp.secs,
                msg.info.header.stamp.nsecs,
            ])
            buffers[topic][counters[topic]] = np.array(msg.data[0].data).reshape(
                msg.data[0].layout.dim[0].size, msg.data[0].layout.dim[1].size
            )

            counters[topic] += 1

        elif topic_type == PointStamped._type:
            # save current msg to be transformed into odom frame whenever a path is published
            curr_goal = copy.deepcopy(msg)

            # save current goal in odom frame
            try:
                transform = tf_buffer.lookup_transform(args.unified_frame, msg.header.frame_id, msg.header.stamp)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr(f"Transform failed of msg {topic_type} at time {t.to_time()}. Disregard message")
                continue

            transformed_goal = tf2_geometry_msgs.do_transform_point(msg, transform)
            buffers[topic][counters[topic]] = np.array([
                transformed_goal.point.x,
                transformed_goal.point.y,
                transformed_goal.point.z,
                msg.header.stamp.secs,
                msg.header.stamp.nsecs,
            ])

            counters[topic] += 1

        elif topic_type == Path._type:
            # save current path (should be in odom frame already)
            for i, pose in enumerate(msg.poses):
                buffers[topic][counters[topic]][i] = np.array([
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                ])

            # save current goal in odom frame
            try:
                transform = tf_buffer.lookup_transform(args.unified_frame, curr_goal.header.frame_id, msg.header.stamp)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr(f"Transform failed of msg {topic_type} at time {t.to_time()}. Disregard message")
                continue

            transformed_goal = tf2_geometry_msgs.do_transform_point(curr_goal, transform)
            metadata[topic][counters[topic]] = np.array([
                transformed_goal.point.x,
                transformed_goal.point.y,
                transformed_goal.point.z,
                msg.poses[0].header.stamp.secs,
                msg.poses[0].header.stamp.nsecs,
            ])

            counters[topic] += 1

        else:
            raise ValueError(f"Topic {topic} is of type {topic_type} which is not supported!")

        # update progress bar
        pbar.update(1)

    bag.close()
    pbar.close()

    # save buffers
    for buffer_name, buffer in buffers.items():
        print(f"Saving {buffer_name} buffer to {buffer_path[buffer_name]}/data.npy")
        np.save(os.path.join(buffer_path[buffer_name], "data.npy"), buffer[: counters[buffer_name]])
    # save metadata
    for buffer_name, buffer in metadata.items():
        print(f"Saving {buffer_name} metadata to {buffer_path[buffer_name]}/metadata.npy")
        np.save(os.path.join(buffer_path[buffer_name], "metadata.npy"), buffer[: counters[buffer_name]])

    print("SUCCESS")
    return


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Extract msgs from a ROS bag.")
    parser.add_argument(
        "-bf",
        "--bag_file",
        default=(
            "/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-14-14-36-02_forest_kaeferberg_entanglement/merged.bag"
        ),
        # default="/media/pascal/T7 Shield/FDMData/2024-09-23-Polyterasse/2024-09-23-10-52-57/fdm_relevant/merged.bag",
        help="Input ROS bag.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/media/pascal/T7 Shield/FDMData/GrandTour/2024-11-14-14-36-02_forest_kaeferberg_entanglement/export",
        help="Output directory.",
    )
    parser.add_argument(
        "-t",
        "--topics",
        nargs="+",
        type=str,
        default=[
            "/elevation_mapping/elevation_map_raw",
            "/anymal_low_level_controller/actuator_readings",
            "/state_estimator/anymal_state",
            "/dlio/lidar_map_odometry",
            "/twist_mux/twist",
        ],  # "/path_planning_and_following/twist", "/clicked_point", "/planner_node/path",
        help="Topics.",
    )
    parser.add_argument(
        "-st", "--start_time", type=float, default=0, help="Start time when messages are read from the rosbag"
    )
    parser.add_argument(
        "-et", "--end_time", type=float, default=None, help="Stop time when messages are read from the rosbag"
    )
    parser.add_argument(
        "-tf",
        "--tf_bags",
        nargs="+",
        type=list,
        # default=[
        #     "/media/pascal/T7 Shield/FDMData/2024-08-14-10-45-39/2024-08-14-10-45-39_nuc_tf_0.bag",
        #     "/media/pascal/T7 Shield/FDMData/2024-08-14-10-45-39/2024-08-14-10-45-39_nuc_tf_1.bag",
        #     "/media/pascal/T7 Shield/FDMData/2024-08-14-10-45-39/2024-08-14-10-45-39_lpc_tf_0.bag",
        #     "/media/pascal/T7 Shield/FDMData/2024-08-14-10-45-39/2024-08-14-10-45-39_lpc_tf_1.bag",
        # ],
        default=None,
        help="Topics.",
    )
    parser.add_argument(
        "--unified_frame",
        type=str,
        default="dlio_map",
        # default="enu_origin",
        help="Frame to transform all messages to.",
    )
    args = parser.parse_args()
    print(args)

    # run main
    main(args)
