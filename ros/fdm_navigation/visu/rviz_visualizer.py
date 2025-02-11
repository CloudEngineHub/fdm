# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from math import cos, sin

import rospy
import seaborn as sns
import tf2_ros
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped
from grid_map_msgs.msg import GridMap, GridMapInfo
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import ColorRGBA, Float32MultiArray, Header, MultiArrayDimension
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray


class NumpyToRviz:
    def __init__(
        self,
        init_node=True,
        cv_bridge=None,
        image_topics=[],
        gridmap_topics=[],
        pointcloud_topics=[],
        camera_info_topics=[],
        path_topics=[],
        marker_topics=[],
        marker_array_topics=[],
        pose_topics=[],
        node_name="numpy_to_rviz",
    ):
        if init_node:
            rospy.init_node(node_name, anonymous=False)

        self.cv_bridge = cv_bridge

        self.pub = {}

        self.register_publisher(image_topics, Image, queue_size=1)
        self.register_publisher(gridmap_topics, GridMap, queue_size=1)
        self.register_publisher(pointcloud_topics, PointCloud2, queue_size=1)
        self.register_publisher(camera_info_topics, CameraInfo, queue_size=1)
        self.register_publisher(path_topics, Path, queue_size=1)
        self.register_publisher(marker_topics, Marker, queue_size=1)
        self.register_publisher(marker_array_topics, MarkerArray, queue_size=1)
        self.register_publisher(pose_topics, PoseStamped, queue_size=1)

        self.br = tf2_ros.TransformBroadcaster()

    def register_publisher(self, topic_list, msg_type, queue_size):
        for topic in topic_list:
            if topic not in self.pub:
                self.pub[topic] = rospy.Publisher(f"~{topic}", msg_type, queue_size=queue_size)
            else:
                raise ValueError(f"Topic {topic} already registered")

    def tf(self, msg, reference_frame="crl_rzr/map"):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = reference_frame
        t.child_frame_id = msg[3]
        t.transform.translation.x = msg[4][0]
        t.transform.translation.y = msg[4][1]
        t.transform.translation.z = msg[4][2]
        t.transform.rotation.x = msg[5][0]
        t.transform.rotation.y = msg[5][1]
        t.transform.rotation.z = msg[5][2]
        t.transform.rotation.w = msg[5][3]
        self.br.sendTransform(t)

    def image(self, topic, img, reference_frame, encoding="bgr8"):
        msg = self.cv_bridge.cv2_to_imgmsg(img, encoding=encoding)
        msg.header.frame_id = reference_frame
        msg.header.stamp = rospy.Time.now()
        self.pub[topic].publish(msg)

    def pointcloud(self, topic, points, reference_frame):
        data = np.zeros(points.shape[0], dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]
        raise ValueError("Fix this")
        # msg = ros_numpy.msgify(PointCloud2, data)
        # msg.header.frame_id = reference_frame

        # self.pub[topic].publish(msg)

    def pose(self, topic, position, orientation, reference_frame):
        msg = PoseStamped()
        msg.header = self.get_header(reference_frame)
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        self.pub[topic].publish(msg)

    def get_header(self, reference_frame):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = reference_frame
        return header

    def marker(self, topic, points, reference_frame, color=(0, 0, 1, 1), msg_type=Marker.LINE_LIST):
        msg = Marker()
        msg.header = self.get_header(reference_frame)
        msg.type = msg_type
        msg.action = Marker.ADD

        msg.pose.position.y = 0
        msg.pose.position.x = 0
        msg.pose.position.z = 0
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.scale.z = 0.05
        msg.lifetime = rospy.Duration(2)
        msg.frame_locked = False

        if len(points.shape) == 3:
            for i in range(points.shape[0]):
                msg.points = []
                mod = points.shape[1] % 2
                for j in range(points.shape[1] - mod):
                    point_msg = Point(x=points[i, j, 0], y=points[i, j, 1], z=points[i, j, 2])
                    msg.points.append(point_msg)
                    # msg.colors.append(ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3]))
                msg.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
                self.pub[topic].publish(msg)

        else:
            mod = points.shape[0] % 2
            for i in range(points.shape[0] - mod):
                msg.points.append(Point(x=points[i, 0], y=points[i, 1], z=points[i, 2]))
            msg.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            self.pub[topic].publish(msg)

    def path(self, topic, path=None, xy_yaw=None, reference_frame=None):
        msg = Path()
        if xy_yaw is None:
            path = path.cpu().numpy()
            for i in range(path.shape[0]):
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = path[i, 0]
                pose_msg.pose.position.y = path[i, 1]
                pose_msg.pose.position.z = path[i, 2]
                pose_msg.pose.orientation.x = 0
                pose_msg.pose.orientation.y = 0
                pose_msg.pose.orientation.z = 0
                pose_msg.pose.orientation.w = 1
                pose_msg.header = self.get_header(reference_frame)
                msg.poses.append(pose_msg)
        elif path is None:
            xy_yaw = xy_yaw.cpu().numpy()
            for i in range(xy_yaw.shape[0]):
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = xy_yaw[i, 0]
                pose_msg.pose.position.y = xy_yaw[i, 1]
                pose_msg.pose.position.z = 0
                yaw = xy_yaw[i, 2]
                quaternion = quaternion_from_euler(0, 0, yaw)  # Roll, pitch, yaw
                pose_msg.pose.orientation.x = quaternion[0]
                pose_msg.pose.orientation.y = quaternion[1]
                pose_msg.pose.orientation.z = quaternion[2]
                pose_msg.pose.orientation.w = quaternion[3]
                pose_msg.header = self.get_header(reference_frame)
                msg.poses.append(pose_msg)

        msg.header = self.get_header(reference_frame)
        self.pub[topic].publish(msg)

    def marker_array(self, topic, path=None, xy_yaw=None, reference_frame=None):
        rospy.init_node("trajectory_publisher", anonymous=True)
        pub = rospy.Publisher("trajectory_arr", MarkerArray, queue_size=10)
        rate = rospy.Rate(20)

        for k in range(20):
            msg = MarkerArray()
            nr = 20
            palette = sns.color_palette("viridis", nr)
            for i in range(nr):
                # Calculate x, y coordinates for a circle trajectory
                x = 5 * cos(i / nr * 3)
                y = 5 * sin(i / nr * 3)

                c = palette[i]

                ma = Marker()
                ma.id = i
                ma.header.frame_id = "odom"
                ma.header.stamp = rospy.Time.now()
                ma.ns = "trajectory"
                ma.type = Marker.MESH_RESOURCE
                ma.mesh_resource = "package://fdm_navigation_ros/meshes/anymal_base.dae"
                ma.mesh_use_embedded_materials = False

                ma.action = Marker.ADD
                ma.scale.x = 1.0
                ma.scale.y = 1.0
                ma.scale.z = 1.0
                ma.pose = Pose(Point(*(x, y, 0)), Quaternion(*(0, 0, 0, 1)))
                ma.color = ColorRGBA(*c, 0.5)

                msg.markers.append(ma)

            pub.publish(msg)
            rate.sleep()

        rospy.init_node("trajectory_publisher", anonymous=True)
        pub = rospy.Publisher("trajectory", Marker, queue_size=10)
        rate = rospy.Rate(10)  # 10Hz

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Diameter of circle
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0

        # Define trajectory points
        num_points = 50
        for i in range(num_points):
            # Calculate x, y coordinates for a circle trajectory
            x = 5 * cos(i * 0.1)
            y = 5 * sin(i * 0.1)

            # Set the position of the circle
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0

            # Calculate color based on time (rainbow color palette)
            color = ColorRGBA()
            color.r = sin(i * 0.1 + 0) * 127 + 128
            color.g = sin(i * 0.1 + 2) * 127 + 128
            color.b = sin(i * 0.1 + 4) * 127 + 128
            color.a = 1.0
            marker.color = color

            # Publish the marker
            marker.header.stamp = rospy.Time.now()
            pub.publish(marker)
            rate.sleep()

    def gridmap(self, topic, msg):
        data_in = msg[0]

        size_x = data_in["data"].shape[1]
        size_y = data_in["data"].shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data = []

        for i in range(data_in["data"].shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = data_in["data"][i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.position.x = data_in["position"][0]
        info.pose.position.y = data_in["position"][1]
        info.pose.position.z = data_in["position"][2]
        info.pose.orientation.x = data_in["orientation_xyzw"][0]
        info.pose.orientation.y = data_in["orientation_xyzw"][1]
        info.pose.orientation.z = data_in["orientation_xyzw"][2]
        info.pose.orientation.w = data_in["orientation_xyzw"][3]
        info.header = msg[1]
        # info.header.stamp.secs = msg[2]vis
        # info.header.stamp = rospy.Time.now()
        info.resolution = data_in["resolution"]
        info.length_x = size_x * data_in["resolution"]
        info.length_y = size_y * data_in["resolution"]

        gm_msg = GridMap(info=info, layers=data_in["layers"], basic_layers=data_in["basic_layers"], data=data)
        self.pub[topic].publish(gm_msg)

    def gridmap_from_numpy(self, topic, data, resolution, layers, reference_frame, x=0, y=0):
        size_x = data.shape[1]
        size_y = data.shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data_out = []

        for i in range(data.shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = data[i, ::-1, ::-1].transpose().ravel()
            data_out.append(data_tmp)

        info = GridMapInfo()
        info.pose.orientation.w = 1
        info.header.seq = 0
        info.header.stamp = rospy.Time.now()
        info.resolution = resolution
        info.length_x = size_x * resolution
        info.length_y = size_y * resolution
        info.header.frame_id = reference_frame
        info.pose.position.x = x
        info.pose.position.y = y
        gm_msg = GridMap(info=info, layers=layers, basic_layers=[], data=data_out)
        self.pub[topic].publish(gm_msg)
        return gm_msg
