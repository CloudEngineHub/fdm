#!/usr/bin/env python3

# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import math
import numpy as np
import os
import torch
import yaml
from scipy.ndimage import distance_transform_edt
from threading import Thread
from typing import Tuple

import hydra
import omegaconf
import pypose as pp
import rospkg
import rospy
import seaborn as sns
import tf2_ros
from anymal_msgs.msg import AnymalState
from dynamic_reconfigure.server import Server
from fdm_navigation_ros.cfg import BaseConfig
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, TwistStamped
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Path
from pytictac import Timer
from series_elastic_actuator_msgs.msg import SeActuatorReadings
from std_msgs.msg import ColorRGBA
from tf2_geometry_msgs import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

import fdm_navigation.helper.math_utils as math_utils
from fdm_navigation.cfg import get_planner_cfg
from fdm_navigation.trajectory_optimizer import SimpleSE2TrajectoryOptimizer

# joint ordering on the real robot
ANYMAL_JOINT_NAMES_REAL = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]
ANYMAL_JOINT_NAMES_SIM = [
    "LF_HAA",
    "LH_HAA",
    "RF_HAA",
    "RH_HAA",
    "LF_HFE",
    "LH_HFE",
    "RF_HFE",
    "RH_HFE",
    "LF_KFE",
    "LH_KFE",
    "RF_KFE",
    "RH_KFE",
]


# Custom constructor for `slice` object
def slice_constructor(loader, node):
    # The node.value will be a list of the slice parameters
    value = loader.construct_sequence(node)
    return slice(*value)


# Add the constructor to the PyYAML loader
yaml.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", slice_constructor, Loader=yaml.FullLoader)


class PlannerNode:
    def __init__(self, node_name):
        """Design Considerations:

        Handling Configurations:
            rosparams
            dynamic_reconfiguration
            default_dataclasses_setup for trajectory optimizer (Dataclasses -> Omegaconfig -> Hydra)
            -> Idea we can potentially modify the Omegaconfiguration using the rosparams and dynamic reconfiguration
            -> Simple overwrite should do the trick and always reinit everything

        Callbacks:
            goal_callback - updates the goal - if goal is new trigger planning - otherwise do nothing
            gridmap_callback - write traversability into observation

        Threads:
            planning_thread - triggers replanning at certain frequency if goal not reached

        Concurrency:
            all threads are executed concurrently therefore no real multi-processing is happening
            lock guard is only implemented in one setting but currently not correct

        Frames:
            all planning is done in odom - this allows for a smooth path in odometry
            the goal is regularly updated from map to odom -> this allows for globally consistent goals
            this setup "low-pass" filters the odometry drift in some sense and avoids that trajectories are jumping

        verify if there is some problem with indexing the gridmap if it is in a different frame the odometry published

        """

        ###
        # Setup ROS Node
        ###

        rospy.init_node(node_name, anonymous=False)

        # Read ROS Params and potentially dynamic reconfigure
        self.read_ros_params()
        if self.use_dynamic_reconfigure:
            self.srv = Server(BaseConfig, self.dynamic_configuration_callback)

        ###
        # FDM Model and config
        ###

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and self.cuda_device_number is not None:
            self.device = f"cuda:{self.cuda_device_number}"
        print(
            f"[INFO] Using device: {self.device} (Device name: {torch.cuda.get_device_name(self.device)}) and compute"
            f" capability: {torch.cuda.get_device_capability(self.device)}"
        )

        # load the jit compiled model
        if "cuda" in self.device:
            self.model = (
                torch.jit.load(os.path.join(self.model_path, "export", "model_cuda_jit.pth")).to(self.device).eval()
            )
        else:
            self.model = torch.jit.load(os.path.join(self.model_path, "export", "model_cpu_jit.pth")).eval()
        self.model = torch.jit.freeze(self.model)
        with open(os.path.join(self.model_path, "params", "config.yaml")) as file:
            self.model_params = yaml.load(file, Loader=yaml.FullLoader)

        ###
        # Initialize the planner
        ###
        if self.restricted_sideward_motion_range is not None:
            vel_lim_y = self.restricted_sideward_motion_range
        else:
            vel_lim_y = self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["lin_vel_y"]
        cfg_dict = get_planner_cfg(
            self.model_params["model_cfg"]["prediction_horizon"],
            "MPPI",
            self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["lin_vel_x"],
            vel_lim_y,
            self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["ang_vel_z"],
            self.debug,
            self.device,
        )

        cfg = omegaconf.OmegaConf.create(cfg_dict)
        self.planner: SimpleSE2TrajectoryOptimizer = hydra.utils.instantiate(cfg.to)

        # height map center
        height_map_center = (
            torch.tensor(self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["size"], device=self.device)
            / 2
            - torch.tensor(self.model_params["env_cfg"]["scene"]["env_sensor"]["offset"]["pos"], device=self.device)[:2]
        )
        self.planner.set_fdm_classes(
            self.model,
            height_map_center=height_map_center,
            height_map_resolution=self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["resolution"],
            height_map_size=self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["size"],
        )

        ###
        # Joint Mapping
        ###

        self.joint_mapping = torch.tensor(
            [ANYMAL_JOINT_NAMES_REAL.index(joint_name) for joint_name in ANYMAL_JOINT_NAMES_SIM]
        )
        self.GRAVITY_VEC_W = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)

        ###
        # RayCaster Pattern
        ###

        self.ray_pattern = math_utils.grid_pattern(
            size=self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["size"],
            resolution=self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["resolution"],
            ordering=self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["ordering"],
            direction=self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["direction"],
            offset=self.model_params["env_cfg"]["scene"]["env_sensor"]["offset"]["pos"],
            device=self.device,
        )
        self.ray_pattern[..., 1] *= -1

        ###
        # Publishers and Subscribers
        ###

        # Output of sampling based optimization
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.path_marker_pub = rospy.Publisher(self.path_topic + "_marker", MarkerArray, queue_size=10)

        # Refine trajectory
        self.optimized_path_pub = rospy.Publisher(self.optimized_path_topic, Path, queue_size=10)

        # twist commands
        self.twist_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        self.smooth_twist_pub = rospy.Publisher(self.smooth_twist_topic, TwistStamped, queue_size=10)
        self.lookahead_twist_pub = rospy.Publisher(self.lookahead_twist_topic, TwistStamped, queue_size=10)

        # Projected Goal
        self.goal_pub = rospy.Publisher(self.projected_goal_topic, PoseStamped, queue_size=10)

        # Listener may be better given that we have to process less odometry messages
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Before registering the callbacks make sure that at least one message of each required topic has been received using the wait_for_message

        print("Verify GridMap is published on topic: ", self.gridmap_topic)
        gridmap_msg = rospy.wait_for_message(self.gridmap_topic, GridMap)
        print("Verify if joint actions are published on topic: ", self.joint_actions_topic)
        rospy.wait_for_message(self.joint_actions_topic, SeActuatorReadings)
        print("Verify if state estimator is published on topic: ", self.state_estimator_topic)
        rospy.wait_for_message(self.state_estimator_topic, AnymalState)
        print("Not verify if goal is published on topic: ", self.goal_topic)

        # Initialize buffers on GPU for constant memory footprint and timings
        self.init_buffers(gridmap_msg)

        self.gridmap_sub = rospy.Subscriber(self.gridmap_topic, GridMap, self.gridmap_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=10)
        self.joint_actions_sub = rospy.Subscriber(
            self.joint_actions_topic, SeActuatorReadings, self.joint_actions_callback, queue_size=1
        )
        self.state_estimator_sub = rospy.Subscriber(
            self.state_estimator_topic, AnymalState, self.state_estimator_callback, queue_size=1
        )

        ###
        # Thread to follow the latest path to increase command frequency
        ###
        if not self.add_path_to_plan or not self.predict_path_without_plan:
            self.tracking_thread = Thread(target=self.tracking_thread_func, name="planning")
            self.tracking_thread.start()

        ###
        # Modification for title fig
        ###
        if self.add_path_to_plan or self.predict_path_without_plan:
            if self.add_path_to_plan:
                self.plot_paths = torch.tensor(
                    [
                        [
                            [0.93, -0.01, -0.09],
                            [0.81, -0.05, -0.06],
                            [1.15, -0.08, -0.01],
                            [0.72, 0.43, 0.12],
                            [0.85, 0.38, 0.23],
                            [1.04, 0.23, 0.53],
                            [1.07, 0.07, 0.43],
                            [0.96, 0.12, 0.47],
                            [1.27, 0.15, 0.21],
                            [1.41, 0.15, 0.04],
                        ],
                        [
                            [1.47, 0.10, -0.68],
                            [1.50, 0.34, -0.75],
                            [1.22, 0.14, -0.51],
                            [1.32, 0.19, -0.32],
                            [1.12, 0.05, -0.08],
                            [1.23, 0.02, 0.02],
                            [1.34, 0.10, 0.08],
                            [1.12, 0.07, 0.16],
                            [1.14, 0.12, 0.28],
                            [1.12, 0.15, 0.53],
                        ],
                        [
                            [1.24, 0.10, -0.40],
                            [1.04, 0.34, -0.35],
                            [1.34, 0.14, -0.41],
                            [1.14, 0.19, -0.53],
                            [1.04, 0.05, -0.62],
                            [1.14, 0.02, -0.81],
                            [1.24, 0.10, -0.91],
                            [1.24, 0.07, -0.97],
                            [1.14, 0.12, -0.92],
                            [1.24, 0.15, -0.87],
                        ],
                        [
                            [1.20, 0.10, 0.2],
                            [1.32, 0.12, 0.2],
                            [1.11, 0.14, 0.2],
                            [1.23, 0.19, 0.0],
                            [1.12, 0.05, 0.0],
                            [1.25, 0.02, -0.2],
                            [1.32, 0.10, -0.3],
                            [1.21, 0.07, 0.0],
                            [1.25, 0.12, 0.2],
                            [1.20, 0.15, 0.2],
                        ],
                        [
                            [1.47, 0.10, 0.12],
                            [1.35, 0.12, 0.02],
                            [1.22, 0.14, 0.41],
                            [1.32, 0.19, 0.73],
                            [1.12, 0.05, 0.22],
                            [1.23, 0.02, 0.03],
                            [1.34, 0.10, 0.34],
                            [1.12, 0.07, 0.47],
                            [1.14, 0.12, 0.06],
                            [1.12, 0.15, -0.24],
                        ],
                        [
                            [1.41, 0.34, 0.35],
                            [1.22, 0.23, 0.41],
                            [1.32, 0.19, 0.53],
                            [1.12, 0.05, 0.62],
                            [1.23, 0.02, 0.81],
                            [1.34, 0.10, 0.91],
                            [1.12, 0.07, 0.79],
                            [1.14, 0.12, 0.58],
                            [1.12, 0.15, 0.73],
                            [1.16, 0.15, 0.82],
                        ],
                        [
                            [1.23, 0.12, 0.64],
                            [1.11, 0.14, 0.44],
                            [0.97, 0.19, 0.52],
                            [1.05, 0.05, 0.18],
                            [0.91, 0.02, -0.2],
                            [0.89, 0.10, -0.3],
                            [0.11, 0.07, 0.01],
                            [1.04, 0.12, 0.21],
                            [1.04, 0.15, 0.21],
                            [1.07, 0.15, 0.21],
                        ],
                        [
                            [1.4438, -6.3849e-03, -0.07],
                            [1.2627, 5.7464e-02, -0.13],
                            [1.3297, 1.0000e-01, -0.18],
                            [1.5000, 1.0648e-03, -0.12],
                            [1.2677, -1.0000e-01, -0.15],
                            [1.4444, -1.0000e-01, -0.32],
                            [1.1133, -1.0000e-01, -0.52],
                            [1.1195, 1.0000e-01, -0.40],
                            [1.0733, 1.0000e-01, -0.32],
                            [1.1061, 1.0000e-01, -0.23],
                        ],
                        [
                            [1.4438e00, -6.3849e-03, 6.6977e-02],
                            [1.2627e00, 5.7464e-02, 1.3386e-01],
                            [1.3297e00, 1.0000e-01, -5.5950e-02],
                            [1.5000e00, 1.0648e-03, 5.4765e-02],
                            [1.2677e00, -1.0000e-01, -4.6979e-02],
                            [1.4444e00, -1.0000e-01, -1.6434e-01],
                            [1.1133e00, -1.0000e-01, 7.6090e-03],
                            [1.1195e00, 1.0000e-01, -1.8821e-01],
                            [1.0733e00, 1.0000e-01, -6.2799e-02],
                            [1.1061e00, 1.0000e-01, -2.4524e-01],
                        ],
                    ],
                    device=self.device,
                )
            else:
                self.plot_paths = torch.tensor(
                    [
                        # center right path: turn right first, then more straight
                        [
                            [0.56, -0.09, -0.32],
                            [0.78, -0.09, -0.45],
                            [0.86, -0.14, -0.32],
                            [0.76, -0.19, -0.15],
                            [0.72, -0.05, 0.02],
                            [0.83, -0.02, 0.23],
                            [0.65, -0.10, 0.21],
                            [0.56, -0.07, 0.01],
                            [0.86, -0.12, 0.05],
                            [1.02, -0.15, 0.01],
                        ],
                        # center left path: turn left first, then more straight
                        [
                            [0.56, 0.09, 0.32],
                            [0.78, 0.09, 0.45],
                            [0.86, 0.14, 0.32],
                            [0.76, 0.19, 0.15],
                            [0.72, 0.05, -0.02],
                            [0.83, 0.02, -0.23],
                            [0.65, 0.10, -0.21],
                            [0.56, 0.07, -0.01],
                            [0.86, 0.12, -0.05],
                            [1.02, 0.15, -0.01],
                        ],
                        # right path: turn right first, then more straight
                        [
                            [0.16, -0.29, -0.23],
                            [0.18, -0.29, -0.35],
                            [0.16, -0.24, -0.23],
                            [0.34, -0.19, -0.10],
                            [0.45, -0.25, 0.02],
                            [0.54, -0.22, 0.23],
                            [0.37, -0.20, 0.21],
                            [0.56, -0.27, 0.01],
                            [0.53, -0.22, 0.05],
                            [0.47, -0.25, 0.01],
                        ],
                        # left path: turn left first, then more straight
                        [
                            [0.16, 0.29, 0.23],
                            [0.18, 0.29, 0.35],
                            [0.16, 0.24, 0.23],
                            [0.34, 0.19, 0.10],
                            [0.45, 0.25, -0.02],
                            [0.54, 0.22, -0.23],
                            [0.37, 0.20, -0.21],
                            [0.56, 0.27, -0.01],
                            [0.53, 0.22, -0.05],
                            [0.47, 0.25, -0.01],
                        ],
                        # straight ahead
                        [
                            [0.86, -6.3849e-03, 6.6977e-02],
                            [0.75, 5.7464e-02, 1.3386e-01],
                            [0.65, 1.0000e-01, -5.5950e-02],
                            [0.71, 1.0648e-03, 5.4765e-02],
                            [0.86, -1.0000e-01, -4.6979e-02],
                            [0.56, -1.0000e-01, -1.6434e-01],
                            [0.67, -1.0000e-01, 7.6090e-03],
                            [0.53, 1.0000e-01, -1.8821e-01],
                            [0.32, 1.0000e-01, -6.2799e-02],
                            [0.21, 1.0000e-01, -2.4524e-01],
                        ],
                    ],
                    device=self.device,
                )

                # init publishers for constant vel
                self.path_pub_const_vel = [
                    rospy.Publisher(self.path_topic + f"_const_vel{i}", Path, queue_size=1)
                    for i in range(self.plot_paths.shape[0])
                ]
                # init publishers for path with high risk
                self.path_pub_high_risk = [
                    rospy.Publisher(self.path_topic + f"_high_risk{i}", Path, queue_size=1)
                    for i in range(self.plot_paths.shape[0])
                ]

            # initialize publishers for additional paths
            self.path_pub_plot = [
                rospy.Publisher(self.path_topic + f"_plt{i}", Path, queue_size=1)
                for i in range(self.plot_paths.shape[0])
            ]

        rospy.loginfo("FDM Planner Ready.")

    """
    Planning
    """

    def spin(self):
        r = rospy.Rate(self.replan_frequency)

        while not rospy.is_shutdown():
            if self.predict_path_without_plan:
                self.plan()
            else:
                self.curr_goal_map.header.stamp = rospy.Time.now()
                suc, new_goal_odom = self.convert_pose_to_se2_in_target_frame(
                    self.curr_goal_map, target_frame=self.odom_frame, limit_to_map=True
                )
                if (
                    suc and self.planner_obs["goal"] is not None
                ):  # and (new_goal_odom != self.planner_obs["goal"]).any():
                    # self.planner_obs["resample_population"] = True
                    self.planner_obs["goal"] = new_goal_odom
                    self.goal_reached = False
                if not self.goal_reached and self.first_goal_received:
                    with Timer("plan time", verbose=self.verbose):
                        self.plan()

            r.sleep()
        rospy.spin()

    def plan(self):
        # Get the current position of the robot
        suc, start = self.get_start_point_se2()

        if suc:
            # save the start position
            self.planner_obs["start"] = start

            # collect proprioceptive and fdm state observations
            self.planner_obs["obs_proprio"] = torch.cat(list(self.proprio_obs.values()), dim=1).unsqueeze(0)
            self.planner_obs["obs_fdm_state"] = math_utils.state_history_transformer(
                self.fdm_state,
                self.model_params["model_cfg"]["history_length"],
            ).unsqueeze(0)

            # get the current height scan measurements
            self.planner_obs["obs_extero"] = (
                self.sample_from_height_scan(self.fdm_state[0].clone()).unsqueeze(0).unsqueeze(0)
            )

            if not self.predict_path_without_plan:
                # project goal onto closest point known point on the height map
                self.publish_goal_msg(self.planner_obs["goal"][0], self.goal_pub)

                # plan the trajectory
                before_store = int(self.planner_obs["resample_population"])
                se2_positions_in_odom, se2_velocity_in_base = self.planner.plan(self.planner_obs, self.height)
                if self.planner_obs["resample_population"] and before_store:
                    self.planner_obs["resample_population"] = False

                # Publish Path
                self.publish_path_msg(se2_positions_in_odom, self.path_pub)
                self.publish_path_msg_as_markers(se2_positions_in_odom, self.path_marker_pub)
                self.se2_positions_in_odom = se2_positions_in_odom

                smoothed_path = pp.chspline(
                    se2_positions_in_odom[0, 0, self.spline_smooth_n :: self.spline_smooth_n, :],
                    interval=1.0 / self.spline_smooth_n,
                )
                self.publish_path_msg(smoothed_path[None, None], self.optimized_path_pub)
                print("Velocity Command:", se2_velocity_in_base[0, 0].cpu().tolist())

                if self.publish_twist and not self.planner.high_risk_path:
                    # Publish Twist
                    self.publish_twist_msg(se2_velocity_in_base[0, 0], self.twist_pub, frame=self.base_frame)
                    smoothed_velocity = pp.chspline(
                        se2_velocity_in_base[0, :: self.spline_smooth_n, :], interval=1.0 / self.spline_smooth_n
                    )
                    self.publish_twist_msg(smoothed_velocity[0], self.smooth_twist_pub, frame=self.base_frame)

                    # start = self.planner_obs["start"]
                    # goal = smoothed_path[self.lookahead_n]
                    # twist = (goal - start) / (self.planner.to_cfg.dt * self.lookahead_n)
                    # # TODO here we should add PD-gains
                    # self.publish_twist_msg(twist[0], self.lookahead_twist_pub, frame=self.odom_frame)
            else:
                self.planner.obs = self.planner_obs
                self.planner.robot_height = self.height

            if self.add_path_to_plan or self.predict_path_without_plan:
                states, risks = self.planner.func(self.plot_paths[:, None, :, :], only_rollout_states=True)
                costs = self.planner.func(self.plot_paths[:, None, :, :])

                if self.predict_path_without_plan:
                    # constant vel paths
                    const_vel_states = self.planner.func(
                        self.plot_paths[:, None, :, :], only_rollout=True, control_mode="velocity_control"
                    )
                    # publish
                    for idx in range(const_vel_states.shape[1]):
                        # constant vel
                        self.publish_path_msg(const_vel_states[0, idx][None, None], self.path_pub_const_vel[idx])
                        if torch.max(risks[idx]) > self.planner.to_cfg.collision_cost_threshold:
                            self.publish_path_msg(states[0, idx][None, None], self.path_pub_high_risk[idx])
                            self.publish_path_msg(states[0, idx][None, None] * 0.0, self.path_pub_plot[idx])
                        else:
                            self.publish_path_msg(states[0, idx][None, None], self.path_pub_plot[idx])
                            self.publish_path_msg(states[0, idx][None, None] * 0.0, self.path_pub_high_risk[idx])

                else:
                    for idx in range(states.shape[1]):
                        self.publish_path_msg(states[0, idx][None, None], self.path_pub_plot[idx])

                print("Publish Paths risks", torch.max(risks, dim=1).values.tolist())
                print("Publish Paths costs", costs.tolist())

    def publish_twist_msg(self, se2_twist_in_odom: torch.Tensor, pub, frame: str, time=None):
        # TODO remove the publishing logic with goal reached to the plan function

        if time is None:
            time = rospy.Time.now()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.odom_frame, rospy.Time(), rospy.Duration(0.1)
            )
        except Exception as e:
            print("Lookup failed:", e)
            return

        p = transform.transform.translation
        q = transform.transform.rotation
        yaw_tf = euler_from_quaternion([q.x, q.y, q.z, q.w], axes="szyx")[0]

        if self.do_not_publish_if_reached:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(0.1)
                )
            except Exception as e:
                print("Lookup failed:", e)
                return

            p = transform.transform.translation
            q = transform.transform.rotation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w], axes="szyx")[0]

            position_offset = torch.norm(torch.tensor([p.x, p.y]) - self.planner_obs["goal"][0, :2].cpu())
            rotation_offset = math_utils.smallest_angle(
                torch.tensor([yaw]), self.planner_obs["goal"][:, 2].cpu()
            ).item()
            if (
                position_offset < self.do_not_publish_distance_offset
                and rotation_offset < self.do_not_publish_heading_offset
            ):
                self.goal_reached = True
                twist_stamped_msg = TwistStamped()
                twist_stamped_msg.header.stamp = time
                twist_stamped_msg.header.frame_id = self.base_frame
                pub.publish(twist_stamped_msg)
                print("Goal Reached - publish 0 twist")
                return

        if frame == self.odom_frame:
            vel_odom = se2_twist_in_odom[:2].cpu()
            rotation_matrix = torch.tensor(
                [[math.cos(yaw_tf), -math.sin(yaw_tf)], [math.sin(yaw_tf), math.cos(yaw_tf)]]
            )
            vel = rotation_matrix @ vel_odom
        elif frame == self.base_frame:
            vel = se2_twist_in_odom[:2].cpu()
        else:
            raise ValueError("Frame not supported")

        # Clip values in base frame
        if self.restricted_forward_motion_range is not None:
            vel[0] = vel[0].clip(self.restricted_forward_motion_range[0], self.restricted_forward_motion_range[1])
        else:
            vel[0] = vel[0].clip(
                self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["lin_vel_x"][0],
                self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["lin_vel_x"][1],
            )

        if self.restricted_sideward_motion_range is not None:
            vel[1] = vel[1].clip(self.restricted_sideward_motion_range[0], self.restricted_sideward_motion_range[1])
        else:
            vel[1] = vel[1].clip(
                self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["lin_vel_y"][0],
                self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["lin_vel_y"][1],
            )

        if self.restricted_rotation_range is not None:
            se2_twist_in_odom[2] = se2_twist_in_odom[2].clip(
                self.restricted_rotation_range[0], self.restricted_rotation_range[1]
            )
        else:
            se2_twist_in_odom[2] = se2_twist_in_odom[2].clip(
                self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["ang_vel_z"][0],
                self.model_params["agent_cfg"]["terms"]["planner"]["agent_term"]["ranges"]["ang_vel_z"][1],
            )

        fac = 1
        if self.invert_twist:
            fac = -1
        twist_stamped_msg = TwistStamped()
        twist_stamped_msg.header.stamp = time
        twist_stamped_msg.twist.linear.x = vel[0].item() * fac
        twist_stamped_msg.twist.linear.y = vel[1].item() * fac
        twist_stamped_msg.twist.angular.z = se2_twist_in_odom[2].item() * fac
        twist_stamped_msg.header.frame_id = self.base_frame

        print("Twist:", vel[0], vel[1], twist_stamped_msg.twist.angular.z)
        pub.publish(twist_stamped_msg)

    def publish_path_msg(self, se2_positions_in_odom, pub, time=None):
        path_msg = Path()
        path_msg.header.frame_id = self.odom_frame

        if time is None:
            time = rospy.Time.now()

        se2_positions_in_odom = se2_positions_in_odom.cpu().numpy()
        for i in range(se2_positions_in_odom.shape[-2]):
            x, y, yaw = se2_positions_in_odom[0, 0, i]
            pose = PoseStamped()
            pose.header.stamp = time
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = self.height
            quaternion = quaternion_from_euler(0, 0, yaw)  # Roll, pitch, yaw
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]

            path_msg.poses.append(pose)

        pub.publish(path_msg)

    def publish_path_msg_as_markers(self, se2_positions_in_odom, pub, time=None):

        msg = MarkerArray()

        if time is None:
            time = rospy.Time.now()

        se2_positions_in_odom = se2_positions_in_odom.cpu().numpy()
        palette = sns.color_palette("viridis", se2_positions_in_odom.shape[-2])

        for i in range(se2_positions_in_odom.shape[-2]):
            x, y, yaw = se2_positions_in_odom[0, 0, i]
            ma = Marker()
            ma.id = i
            ma.header.frame_id = self.odom_frame
            ma.header.stamp = time
            ma.ns = "trajectory"
            ma.type = Marker.MESH_RESOURCE
            ma.mesh_resource = "package://fdm_navigation_ros/meshes/anymal_base.dae"
            ma.mesh_use_embedded_materials = False

            ma.action = Marker.ADD
            ma.scale.x = 1.0
            ma.scale.y = 1.0
            ma.scale.z = 1.0
            quaternion = quaternion_from_euler(0, 0, yaw)  # Roll, pitch, yaw
            ma.pose = Pose(Point(*(x, y, self.height)), Quaternion(*quaternion))
            c = palette[i]
            ma.color = ColorRGBA(*c, 0.2)
            msg.markers.append(ma)

        pub.publish(msg)

    def publish_goal_msg(self, goal, pub):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.odom_frame
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = goal[0].item()
        pose_msg.pose.position.y = goal[1].item()
        pose_msg.pose.position.z = self.height

        quaternion = quaternion_from_euler(0, 0, goal[2].item())  # Roll, pitch, yaw
        pose_msg.pose.orientation = Quaternion(*quaternion)

        pub.publish(pose_msg)

    def read_ros_params(self):
        # TODO what parameters are needed - remove all default values when finished development
        self.debug = rospy.get_param("~debug")
        self.verbose = rospy.get_param("~verbose")

        self.path_topic = rospy.get_param("~path_topic")
        self.optimized_path_topic = rospy.get_param("~optimized_path_topic")

        self.twist_topic = rospy.get_param("~twist_topic")
        self.smooth_twist_topic = rospy.get_param("~smooth_twist_topic")
        self.lookahead_twist_topic = rospy.get_param("~lookahead_twist_topic")

        self.projected_goal_topic = rospy.get_param("~projected_goal_topic")

        self.replan_frequency = rospy.get_param("~replan_frequency")

        self.goal_topic = rospy.get_param("~goal_topic")
        self.gridmap_layer = rospy.get_param("~gridmap_layer")

        # observation space topics
        self.joint_actions_topic = rospy.get_param("~joint_actions_topic")
        self.state_estimator_topic = rospy.get_param("~state_estimator_topic")
        self.gridmap_topic = rospy.get_param("~gridmap_topic")

        # model_path
        self.model_path = rospy.get_param("~model_path")
        self.remove_torque = rospy.get_param("~remove_torque")

        # frames
        self.base_frame = rospy.get_param("~base")
        self.odom_frame = rospy.get_param("~odom")
        self.map_frame = rospy.get_param("~map")

        # Could exist a better integration via dynamic reconfiguration
        self.use_dynamic_reconfigure = rospy.get_param("~use_dynamic_reconfigure")
        self.invert_twist = rospy.get_param("~invert_twist")
        self.publish_twist = rospy.get_param("~publish_twist")
        self.spline_smooth_n = rospy.get_param("~spline_smooth_n")
        self.lookahead_n = rospy.get_param("~lookahead_n")
        self.do_not_publish_if_reached = rospy.get_param("~do_not_publish_if_reached")
        self.do_not_publish_distance_offset = rospy.get_param("~do_not_publish_distance_offset")
        self.do_not_publish_heading_offset = rospy.get_param("~do_not_publish_heading_offset")

        # options to change the planned paths
        self.restricted_forward_motion_range = rospy.get_param("~restricted_forward_motion_range", default=None)
        self.restricted_sideward_motion_range = rospy.get_param("~restricted_sideward_motion_range", default=None)
        self.restricted_rotation_range = rospy.get_param("~restricted_rotation_range", default=None)
        self.height_scan_dilution = rospy.get_param("~height_scan_dilution", default=None)

        # plot params
        self.add_path_to_plan = rospy.get_param("~add_path_to_plan", default=False)
        self.predict_path_without_plan = rospy.get_param("~predict_path_without_plan", default=False)

        # cuda device number
        self.cuda_device_number = rospy.get_param("~cuda_device_number", default=0)

    @torch.inference_mode()
    def dynamic_configuration_callback(self, cfg, level):
        group_dict = cfg["groups"]["groups"]

        if group_dict["general"]["parameters"]["debug"] and not self.planner.to_cfg.init_debug:
            group_dict["general"]["parameters"]["debug"] = False

        # Verify the risk thresholds ordering
        if group_dict["pp"]["parameters"]["risky_th"] < group_dict["pp"]["parameters"]["safe_th"]:
            EPS = 0.0001
            group_dict["pp"]["parameters"]["risky_th"] = group_dict["pp"]["parameters"]["safe_th"] + EPS

        if group_dict["pp"]["parameters"]["fatal_th"] < group_dict["pp"]["parameters"]["risky_th"]:
            group_dict["pp"]["parameters"]["fatal_th"] = group_dict["pp"]["parameters"]["risky_th"]

        for group_name, v in group_dict.items():
            for p_name, _ in v["parameters"].items():
                cfg["groups"]["groups"][group_name]["parameters"][p_name] = cfg[p_name]
                print("updating things")
                value = cfg[p_name]
                if group_name == "general":
                    setattr(self.planner.to_cfg, p_name, value)
                elif group_name == "planner_node":
                    setattr(self, p_name, value)
                else:
                    setattr(self.planner.to_cfg, group_name + "_" + p_name, value)

        print("Dynamicially reconfigured !!!! ")
        return cfg

    """
    High frequency tracking thread
    """

    @torch.inference_mode()
    def tracking_thread_func(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.goal_reached and self.se2_positions_in_odom is not None and not self.planner.high_risk_path:
                # Get the current position of the robot
                suc, start = self.get_start_point_se2()

                # TODO finish this nicely

                # Compute the distance to each path point
                distances = torch.norm(self.se2_positions_in_odom[0, 0, :, :2] - start[0, :2], dim=1)

                # Find the index of the closest path point
                closest_idx = torch.argmin(distances)

                m = torch.ones_like(distances).type(torch.bool)
                m[:closest_idx] = False
                m[distances < 0.3] = False

                if m.max() == 1:
                    closest_idx2 = torch.min(torch.arange(distances.shape[0], device=m.device)[m])
                else:
                    closest_idx2 = distances.shape[0] - 1

                # Determine the lookahead index, ensuring it does not exceed the list bounds
                lookahead_steps = 1  # for example
                target_idx = max(closest_idx + lookahead_steps, closest_idx2)
                target_idx = min(target_idx, self.se2_positions_in_odom.shape[2] - 1)

                # Target position
                target_pos = self.se2_positions_in_odom[0, 0, target_idx]
                print("SE@ PRED ODOM", self.se2_positions_in_odom)
                print("START", start[0])
                twist = target_pos - start[0]
                print("Target Pos:", twist)

                twist[:2] /= torch.norm(twist[:2])
                twist[2] = (twist[2] + torch.pi) % (2 * torch.pi) - torch.pi

                # TODO here we should add PD-gains
                self.publish_twist_msg(twist, self.lookahead_twist_pub, frame=self.odom_frame)

            rate.sleep()

    """
    Buffers
    """

    def init_buffers(self, gridmap_msg: GridMap):
        BS = 1
        self.goal_reached = True
        self.curr_goal_map = PoseStamped()
        self.curr_goal_map.header.frame_id = self.map_frame
        self.curr_goal_map.pose = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))
        self.first_goal_received = False
        self.planner_obs = {
            "goal": torch.zeros((BS, 3), device=self.device),
            "start": torch.zeros((BS, 3), device=self.device),
            "resample_population": False,
        }
        # proprioceptive observations ordered as in the sim observation space
        self.proprio_obs = {
            "commands": torch.zeros((self.model_params["model_cfg"]["history_length"], 3), device=self.device),
            "projection_gravity": torch.zeros(
                (self.model_params["model_cfg"]["history_length"], 3), device=self.device
            ),
            "base_lin_vel": torch.zeros((self.model_params["model_cfg"]["history_length"], 3), device=self.device),
            "base_ang_vel": torch.zeros((self.model_params["model_cfg"]["history_length"], 3), device=self.device),
            "joint_torque": torch.zeros(
                (self.model_params["model_cfg"]["history_length"], len(ANYMAL_JOINT_NAMES_SIM)), device=self.device
            ),
            "joint_pos": torch.zeros(
                (self.model_params["model_cfg"]["history_length"], len(ANYMAL_JOINT_NAMES_SIM)), device=self.device
            ),
            "joint_vel": torch.zeros(
                (self.model_params["model_cfg"]["history_length"], len(ANYMAL_JOINT_NAMES_SIM)), device=self.device
            ),
            "actions": torch.zeros(
                (self.model_params["model_cfg"]["history_length"], len(ANYMAL_JOINT_NAMES_SIM)), device=self.device
            ),
            "prev_actions": torch.zeros(
                (self.model_params["model_cfg"]["history_length"], len(ANYMAL_JOINT_NAMES_SIM)), device=self.device
            ),
        }
        if self.remove_torque:
            self.proprio_obs.pop("joint_torque")
        # fdm state observation space
        # -- [History Length, 3 (pos) + 4 (yaw) + 1 (collision) + 1 (energy)]
        self.fdm_state = torch.zeros((self.model_params["model_cfg"]["history_length"], 9), device=self.device)
        # exteroceptive observations
        self.extero_pos = torch.zeros((3,), device=self.device)
        self.elevation_map_resolution = gridmap_msg.info.resolution
        self.elevation_map_shape = (
            int(gridmap_msg.info.length_x / gridmap_msg.info.resolution),
            int(gridmap_msg.info.length_y / gridmap_msg.info.resolution),
        )
        self.extero_obs = torch.zeros(*self.elevation_map_shape, device=self.device)
        self.extero_dims_low = (
            -np.array(self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["size"]) / 2
            + np.array(self.model_params["env_cfg"]["scene"]["env_sensor"]["offset"]["pos"])[:2]
        )
        self.extero_dims_high = (
            np.array(self.model_params["env_cfg"]["scene"]["env_sensor"]["pattern_cfg"]["size"]) / 2
            + np.array(self.model_params["env_cfg"]["scene"]["env_sensor"]["offset"]["pos"])[:2]
        )

        # init the time buffers to allow for time constraint filtering
        self.last_state_estimator_msg_time = None  # rospy.Time.now().to_sec()
        self.last_joint_actions_msg_time = None  # rospy.Time.now().to_sec()
        self.proprio_msg_min_time_diff = (
            self.model_params["model_cfg"]["command_timestep"] / self.model_params["model_cfg"]["history_length"]
        ) * 0.9

        # for twist thread
        self.goal_reached = True
        self.se2_positions_in_odom = None

        # current height of the robot
        self.height = 0.0

    """
    Subscriber Callbacks
    """

    @torch.inference_mode()
    def gridmap_callback(self, gridmap_msg: GridMap):
        """Callback for the gridmap message and sample the exteroceptive measurepoints from the elevation map."""
        assert self.gridmap_layer in gridmap_msg.layers, "Elevation layer not in gridmap message"

        # get layer idx
        idx = gridmap_msg.layers.index(self.gridmap_layer)
        # Extract grid_map layer as numpy array
        data_list = np.array(gridmap_msg.data[idx].data).copy()
        layout_info = gridmap_msg.data[idx].layout
        n_cols = layout_info.dim[0].size
        n_rows = layout_info.dim[1].size

        # remove nan values
        data_list = np.nan_to_num(
            data_list, copy=False, nan=self.model_params["env_cfg"]["scene"]["env_sensor"]["max_distance"]
        )

        # get height = robot_height - hit_point_z - offset
        data_list += (
            self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"]["offset"]
            - self.fdm_state[0, 2].cpu().numpy()
        )

        # apply an median filter to the height map
        # data_list = median_filter(data_list, (5, 5))
        data_list = data_list.reshape(n_rows, n_cols)

        # apply clipping and generate mask for nearest neighbor filling
        if self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["clip"] is not None:
            data_list = np.clip(
                data_list,
                a_min=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["clip"][0],
                a_max=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["clip"][1],
            )
            mask = (
                data_list
                == self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["clip"][1]
            )
        elif (
            "clip_height"
            in self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"]
            and self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"][
                "clip_height"
            ]
            is not None
        ):
            data_list = np.clip(
                data_list,
                a_min=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"][
                    "clip_height"
                ][0],
                a_max=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"][
                    "clip_height"
                ][1],
            )
            mask = (
                data_list
                == self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"][
                    "clip_height"
                ][1]
            )
        elif (
            "fill_value" in self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"]
            and self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"][
                "fill_value"
            ]
            is not None
        ):
            data_list = np.clip(
                data_list,
                a_min=-1.0,
                a_max=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"][
                    "fill_value"
                ],
            )
            mask = (
                data_list
                == self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"][
                    "fill_value"
                ]
            )
        else:
            data_list = np.clip(
                data_list,
                a_min=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"][
                    "clip_values"
                ][0],
                a_max=self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"][
                    "clip_values"
                ][1],
            )
            mask = (
                data_list
                == self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["noise"][
                    "clip_values"
                ][1]
            )

        # apply nearest neighbor filling
        _, indices = distance_transform_edt(mask, return_indices=True)
        data_list = data_list[tuple(indices)]

        self.extero_mask = torch.tensor(mask, device=self.device)
        self.extero_obs[:, :] = torch.tensor(data_list, dtype=torch.float32, device=self.device)
        self.extero_pos = torch.tensor(
            [gridmap_msg.info.pose.position.x, gridmap_msg.info.pose.position.y, gridmap_msg.info.pose.position.z],
            device=self.device,
        )

        assert not torch.isnan(self.extero_obs).any(), "Nan in height map"

    @torch.inference_mode()
    def goal_callback(self, pose_stamped_msg: PoseStamped):
        rospy.loginfo("Received new goal pose")
        self.curr_goal_map = pose_stamped_msg
        suc, new_goal_odom = self.convert_pose_to_se2_in_target_frame(
            pose_stamped_msg, target_frame=self.odom_frame, limit_to_map=False
        )
        print("Received new goal at: ", new_goal_odom)
        if suc and (new_goal_odom != self.planner_obs["goal"]).any():
            # New goal detected
            self.planner_obs["resample_population"] = True
            self.planner_obs["goal"] = new_goal_odom
            self.goal_reached = False
        # indicator to start the planning
        self.first_goal_received = True

    @torch.inference_mode()
    def joint_actions_callback(self, joint_actions_msg: SeActuatorReadings):
        # check if sufficient time has passed
        if (
            self.last_joint_actions_msg_time is not None
            and joint_actions_msg.readings[0].header.stamp.to_sec() - self.last_joint_actions_msg_time
            < self.proprio_msg_min_time_diff
        ):
            return
        elif False:  # self.verbose:
            print(
                "New joint actions message received after ",
                joint_actions_msg.readings[0].header.stamp.to_sec() - self.last_joint_actions_msg_time,
                " seconds",
            )
        self.last_joint_actions_msg_time = joint_actions_msg.readings[0].header.stamp.to_sec()

        self.proprio_obs["prev_actions"] = torch.roll(self.proprio_obs["prev_actions"], shifts=1, dims=0)
        self.proprio_obs["prev_actions"][0, :] = self.proprio_obs["actions"][
            0, :
        ].clone()  # TODO: check if clone is necessary

        self.proprio_obs["actions"] = torch.roll(self.proprio_obs["actions"], shifts=1, dims=0)
        self.proprio_obs["actions"][0, :] = torch.tensor(
            [single_reading.state.joint_position for single_reading in joint_actions_msg.readings], device=self.device
        )

    @torch.inference_mode()
    def state_estimator_callback(self, state_estimator_msg: AnymalState):
        # check if sufficient time has passed
        if (
            self.last_state_estimator_msg_time is not None
            and state_estimator_msg.header.stamp.to_sec() - self.last_state_estimator_msg_time
            < self.proprio_msg_min_time_diff
        ):
            return
        elif False:
            print(
                "New state estimator message received after ",
                state_estimator_msg.header.stamp.to_sec() - self.last_state_estimator_msg_time,
                " seconds",
            )
        self.last_state_estimator_msg_time = state_estimator_msg.header.stamp.to_sec()

        # get current orientation of the robot
        base_rot = torch.tensor(
            [
                state_estimator_msg.pose.pose.orientation.w,
                state_estimator_msg.pose.pose.orientation.x,
                state_estimator_msg.pose.pose.orientation.y,
                state_estimator_msg.pose.pose.orientation.z,
            ],
            device=self.device,
        )

        # proprioceptive observations
        self.proprio_obs["joint_pos"] = torch.roll(self.proprio_obs["joint_pos"], shifts=1, dims=0)
        self.proprio_obs["joint_pos"][0, :] = torch.tensor(state_estimator_msg.joints.position, device=self.device)[
            self.joint_mapping
        ]

        self.proprio_obs["joint_vel"] = torch.roll(self.proprio_obs["joint_vel"], shifts=1, dims=0)
        self.proprio_obs["joint_vel"][0, :] = torch.tensor(state_estimator_msg.joints.velocity, device=self.device)[
            self.joint_mapping
        ]

        if not self.remove_torque:
            self.proprio_obs["joint_torque"] = torch.roll(self.proprio_obs["joint_torque"], shifts=1, dims=0)
            self.proprio_obs["joint_torque"][0, :] = torch.tensor(
                state_estimator_msg.joints.effort, device=self.device
            )[self.joint_mapping]

        self.proprio_obs["base_lin_vel"] = torch.roll(self.proprio_obs["base_lin_vel"], shifts=1, dims=0)
        self.proprio_obs["base_lin_vel"][0, 0] = state_estimator_msg.twist.twist.linear.x
        self.proprio_obs["base_lin_vel"][0, 1] = state_estimator_msg.twist.twist.linear.y
        self.proprio_obs["base_lin_vel"][0, 2] = state_estimator_msg.twist.twist.linear.z

        self.proprio_obs["base_ang_vel"] = torch.roll(self.proprio_obs["base_ang_vel"], shifts=1, dims=0)
        self.proprio_obs["base_ang_vel"][0, 0] = state_estimator_msg.twist.twist.angular.x
        self.proprio_obs["base_ang_vel"][0, 1] = state_estimator_msg.twist.twist.angular.y
        self.proprio_obs["base_ang_vel"][0, 2] = state_estimator_msg.twist.twist.angular.z

        self.proprio_obs["projection_gravity"] = torch.roll(self.proprio_obs["projection_gravity"], shifts=1, dims=0)
        self.proprio_obs["projection_gravity"][:] = math_utils.quat_rotate_inverse(
            base_rot.unsqueeze(0), self.GRAVITY_VEC_W
        )

        # fdm state observation space
        self.fdm_state = torch.roll(self.fdm_state, shifts=1, dims=0)
        self.fdm_state[0, 0] = state_estimator_msg.pose.pose.position.x
        self.fdm_state[0, 1] = state_estimator_msg.pose.pose.position.y
        self.fdm_state[0, 2] = state_estimator_msg.pose.pose.position.z
        self.fdm_state[0, 3:7] = base_rot
        self.fdm_state[0, 7] = 0.0  # no collision
        if not self.remove_torque:
            self.fdm_state[0, 8] = (
                torch.sum(self.proprio_obs["joint_torque"] ** 2)
                * self.model_params["env_cfg"]["observations"]["fdm_state"]["hard_contact"]["params"][
                    "energy_scale_factor"
                ]
            )
        else:
            self.fdm_state[0, 8] = (
                torch.sum(torch.tensor(state_estimator_msg.joints.effort, device=self.device) ** 2)
                * self.model_params["env_cfg"]["observations"]["fdm_state"]["hard_contact"]["params"][
                    "energy_scale_factor"
                ]
            )

    """
    Helper Functions
    """

    def convert_pose_to_se2_in_target_frame(
        self, pose: PoseStamped, target_frame: str, limit_to_map: bool = False
    ) -> Tuple[bool, torch.Tensor]:
        # transform pose if necessary
        if limit_to_map:

            if pose.header.frame_id != "base":
                try:
                    # tf_to_target = self.tf_buffer.lookup_transform("base", transform.header.frame_id, time=transform.header.stamp)
                    # transform = tf2_geometry_msgs.do_transform_pose(transform, tf_to_target)
                    pose = self.tf_buffer.transform(pose, "base", timeout=rospy.Duration(0.02))
                except Exception as e:
                    print("Error transforming PoseStamped message:", e)
                    return False, None  # Handle the error according to your needs

            # limit the goal to the elevation map
            pose.pose.position.x = np.clip(pose.pose.position.x, self.extero_dims_low[0], self.extero_dims_high[0])
            pose.pose.position.y = np.clip(pose.pose.position.y, self.extero_dims_low[1], self.extero_dims_high[1])

        if pose.header.frame_id != target_frame:
            # transform to target frame
            try:
                pose = self.tf_buffer.transform(pose, target_frame, timeout=rospy.Duration(0.02))
            except Exception as e:
                print("Error transforming PoseStamped message:", e)
                return False, None

        yaw = euler_from_quaternion(
            [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],
            axes="szyx",
        )[0]

        return True, torch.tensor([[pose.pose.position.x, pose.pose.position.y, yaw]], device=self.device)

    def get_start_point_se2(self) -> Tuple[bool, torch.Tensor]:
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(1.0)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f"Lookup transformation {self.base_frame} to {self.odom_frame} failed! {e}")
            return False, None

        p = transform.transform.translation
        q = transform.transform.rotation

        # save height for msg publishing
        self.height = p.z

        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w], axes="szyx")[0]
        return True, torch.tensor([[p.x, p.y, yaw]], device=self.device)

    # @torch.jit.script
    def sample_from_height_scan(self, obs_fdm_state: torch.Tensor) -> torch.Tensor:
        sample_points_odom = math_utils.quat_apply_yaw(obs_fdm_state[3:7], self.ray_pattern)
        # sample_points_odom = math_utils.quat_apply_yaw(math_utils.quat_inv(obs_fdm_state[3:7]), self.ray_pattern)
        sample_points_odom += obs_fdm_state[:3] - self.extero_pos
        # -- convert to idx by diving through the resolution of the elevation map
        sample_idx = torch.round(sample_points_odom[:, :2] / self.elevation_map_resolution).to(torch.int)
        # -- indexes assume middle as center, correct for it
        sample_idx += (torch.tensor(self.elevation_map_shape, device=self.device) - 1) // 2
        # -- clip the indexes to the elevation map size
        sample_idx = torch.clip(sample_idx, 0, self.elevation_map_shape[0] - 1)
        # -- need to rotate the elevation map by 90 degrees
        elevation_map = torch.rot90(self.extero_obs, 1, dims=(0, 1))
        # -- flip along y-axis
        elevation_map = torch.flip(elevation_map, dims=(1,))

        if self.height_scan_dilution:
            print("before", torch.where(elevation_map == torch.max(elevation_map))[0].sum())
            elevation_map = (
                torch.nn.functional.max_pool2d(
                    elevation_map.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2
                )
                .squeeze(0)
                .squeeze(0)
            )
            print("after", torch.where(elevation_map == torch.max(elevation_map))[0].sum())

        # -- get the elevation map at the sample points
        observations_exteroceptive = elevation_map[sample_idx[..., 0], sample_idx[..., 1]]
        # -- reshape into 2d pattern
        observations_exteroceptive = torch.unflatten(
            observations_exteroceptive,
            0,
            self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"]["shape"],
        )

        # if debug visualization is on
        if self.debug:
            # visualize the sampled points of the gridmap
            points = sample_points_odom + obs_fdm_state[:3]
            points[:, 2] = (
                observations_exteroceptive.flatten()
                + obs_fdm_state[2]
                - self.model_params["env_cfg"]["observations"]["fdm_obs_exteroceptive"]["env_sensor"]["params"][
                    "offset"
                ]
            )
            points[:, 2] += 0.05  # add a small offset to see the points

            # debug
            self.planner.ntr.marker(
                "height_map_debug_cb",
                points.cpu().numpy(),
                reference_frame=self.planner.frame_id,
                color=(0.247, 0.349, 0.478, 1.0),
                msg_type=Marker.POINTS,
            )

        return observations_exteroceptive


def reload_rosparams(enabled, node_name):
    if enabled:
        with contextlib.suppress(Exception):
            rospy.delete_param(node_name)

        rospack = rospkg.RosPack()
        simple_nav_path = rospack.get_path("fdm_navigation_ros")
        os.system(f"rosparam load {simple_nav_path}/config/default.yaml {node_name}")


if __name__ == "__main__":
    node_name = "planner_node"

    with torch.inference_mode():
        reload_rosparams(True, node_name)
        node = PlannerNode(node_name)
        node.spin()
