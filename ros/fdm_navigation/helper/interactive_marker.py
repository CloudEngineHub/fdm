# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional

import rospy
from geometry_msgs.msg import Pose
from interactive_markers.interactive_marker_server import InteractiveMarker, InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import InteractiveMarkerControl, Marker


class GenericInteractiveMarker:
    def __init__(self, name: str, frame_id: str, scale: float, menu_config: list = []):

        self.server = InteractiveMarkerServer("server_" + name)
        self.menu_handle = MenuHandler()
        self.menu_config = menu_config

        self._default_frame_id = frame_id
        self._default_name = name
        self._default_color = (1.0, 1.0, 0.0, 1.0)
        self._id = 0
        self._default_scale = scale

        for name in menu_config:
            pose = dir(self).index(f"cb_{name}")
            assert pose != -1, f"defined method {name} not implemented in class"
            self.menu_handle.insert(name, callback=getattr(self, f"cb_{name}"))

    def get_marker(self, scale: float, color: tuple, shape, mesh: Optional[str]):
        marker = Marker()
        marker.type = shape

        if shape == Marker.MESH_RESOURCE:
            marker.mesh_resource = mesh
            marker.mesh_use_embedded_materials = True
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1  # 0.7071068
            marker.pose.orientation.x = 0.0  # 0.7071068
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
        else:
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            marker.scale.x = scale * 0.75
            marker.scale.y = scale * 0.75
            marker.scale.z = scale * 0.75

        # self._marker_list.append(marker)
        return marker

    def get_int_marker(self, pose: Pose, scale: float, color: tuple, shape=Marker.SPHERE, mesh=None, name=None):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._default_frame_id
        # int_marker.header.stamp = rospy.Time.now()
        # int_marker.pose = pose
        int_marker.pose = pose
        int_marker.scale = scale

        if not (name is None):
            int_marker.name = name
        else:
            int_marker.name = f"{self._default_name}_{self._id}"

        int_marker.description = str(int_marker.name)  # "No description"

        # self._int_marker_list.append(int_marker)
        return int_marker

    def add_marker(
        self,
        pose: Pose,
        cb_name: str,
        scale: Optional[float] = None,
        color: Optional[tuple] = None,
        shape=Marker.SPHERE,
        mesh=None,
        name=None,
    ):
        if color is None:
            color = self._default_color

        if name is None:
            self._id += 1

        if scale is None:
            scale = self._default_scale

        int_marker = self.get_int_marker(pose, scale, color, shape, mesh, name=name)
        marker = self.get_marker(scale, color, shape, mesh)

        self.add_control_to_marker(int_marker, marker)

        #  self.add_menu_to_marker(marker, box)
        self.server.insert(int_marker, getattr(self, cb_name))
        self.menu_handle.apply(self.server, int_marker.name)
        self.server.applyChanges()
        print("Marker ID ", self._id)

    def add_control_to_marker(self, int_marker, marker):
        control = InteractiveMarkerControl()
        if len(self.menu_config) > 0:
            # Add interactive marker
            control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = math.sqrt(2)
        control.orientation.x = 0
        control.orientation.y = math.sqrt(2)
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = math.sqrt(2)
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = math.sqrt(2)
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = math.sqrt(2)
        control.orientation.x = math.sqrt(2)
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)


if __name__ == "__main__":
    # Example usage

    rospy.init_node("server_simple_marker")

    class MainMarkerServer(GenericInteractiveMarker):
        def __init__(self, name: str, frame_id: str, menu_config: list = []):
            super().__init__(name, frame_id, menu_config)

        def cb_move(self, feedback):
            p = feedback.pose.position
            print(feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z))
            # TODO query elevation map and set the Z-position

        def cb_set_pose(self, feedback):
            print("set_pose")

        def cb_save_path(self, feedback):
            print("save_path")

        def cb_reset_path(self, feedback):
            print("reset_path")

    class PathMarkerServer(GenericInteractiveMarker):
        def __init__(self, name: str, frame_id: str, menu_config: list = []):
            super().__init__(name, frame_id, menu_config)

        def cb_move(self, feedback):
            p = feedback.pose.position
            print(feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z))

        def cb_update_pose(self, feedback):
            print("update_pose")

        def cb_delete_pose(self, feedback):
            print("delete_pose")

    path_marker_server = PathMarkerServer(
        name="path_marker", frame_id="base_link", menu_config=["update_pose", "delete_pose"]
    )
    main_marker_server = MainMarkerServer(
        name="main_marker", frame_id="base_link", menu_config=["set_pose", "save_path", "reset_path"]
    )

    p = Pose()
    p.position.x = 0
    p.position.x = 0
    p.position.x = 0
    p.orientation.x = 0
    p.orientation.y = 0
    p.orientation.z = 0
    p.orientation.w = 1

    path_marker_server.add_marker(
        pose=p,
        cb_name="cb_move",
        scale=0.4,
        color=(209.0 / 255, 134.0 / 255, 0.0 / 255, 1),
    )

    main_marker_server.add_marker(
        pose=p,
        cb_name="cb_move",
        scale=0.5,
        color=(15.0 / 255, 122.0 / 255, 175.0 / 255, 1),
    )

    path_marker_server.add_marker(
        pose=p,
        cb_name="cb_move",
        scale=0.4,
        color=(209.0 / 255, 134.0 / 255, 0.0 / 255, 1),
    )

    main_marker_server.add_marker(
        pose=p,
        cb_name="cb_move",
        scale=0.5,
        color=(15.0 / 255, 122.0 / 255, 175.0 / 255, 1),
    )

    rospy.spin()
