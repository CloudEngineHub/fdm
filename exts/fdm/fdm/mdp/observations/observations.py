# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ActuatorNetMLP
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, MultiMeshRayCaster, RayCaster, RayCasterCamera
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_dynamic_meshes

from nav_tasks.mdp import GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from fdm.mdp import MixedCommand, NavigationSE2Action

"""
Root state.
"""


def base_orientation_xyzw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Orientation of the asset's root in world frame.

    Note: converts the quaternion to (x, y, z, w) format."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w[:, [1, 2, 3, 0]]


def base_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Position of the asset's root in world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def joint_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def joint_pos_error_history(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_idx: int = 0
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    actuators: ActuatorNetMLP = env.scene[asset_cfg.name].actuators["legs"]
    return actuators._joint_pos_error_history[:, history_idx]


def joint_velocity_history(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_idx: int = 0
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    actuators: ActuatorNetMLP = env.scene[asset_cfg.name].actuators["legs"]
    return actuators._joint_vel_history[:, history_idx]


class FrictionObservation:
    """Friction observation."""

    def __init__(self):
        pass

    def _setup_view(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        self._num_shapes_per_body_mapping = []

        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            self._num_shapes_per_body_mapping.append(link_physx_view.max_shapes)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The friction coefficients of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # setup the view
        if not hasattr(self, "_num_shapes_per_body_mapping"):
            self._setup_view(env, asset_cfg)

        # get t materials of the bodies
        materials = asset.root_physx_view.get_material_properties()
        static_friction = torch.zeros(
            (env.num_envs, len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies, 1),
            device=env.device,
        )

        # get material properties for the bodies
        for idx, body_id in enumerate(
            asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else range(asset.num_bodies)
        ):
            # start index of shape
            start_idx = sum(self._num_shapes_per_body_mapping[:body_id])
            # end index of shape
            end_idx = start_idx + self._num_shapes_per_body_mapping[body_id]
            # get the static friction
            if end_idx - start_idx > 1:
                static_friction[:, idx, 0] = materials[:, start_idx:end_idx, 0].mean(dim=-1)
            else:
                static_friction[:, idx] = materials[:, start_idx:end_idx, 0]

        return static_friction.squeeze(-1)

    def __name__(self):
        return "FrictionObservation"


def se2_root_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The root position of the asset in the SE(2) frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get yaw angle of the root
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    # return the root position in the SE(2)
    return torch.cat([asset.data.root_pos_w[:, :2], yaw.unsqueeze(-1)], dim=-1)


"""
Sensors
"""


def height_scan_bounded(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - 0.5
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # assign max distance to inf values
    height[torch.isinf(height)] = sensor.cfg.max_distance
    height[torch.isnan(height)] = sensor.cfg.max_distance
    return height


def lidar2Dnormalized(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Lidar scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # return the height scan
    distances = torch.norm((sensor.data.ray_hits_w - sensor.data.pos_w[:, None, :]), dim=-1)
    # clip inf values to max_distance
    distances[torch.isinf(distances)] = sensor.cfg.max_distance
    # returned clipped to the sensor's range
    return torch.clip(distances, 0.0, sensor.cfg.max_distance) / sensor.cfg.max_distance


def raycast_depth_camera_data(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    """Images generated by the raycast camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # return the data
    output = sensor.data.output[data_type].clone().unsqueeze(-1)
    output[torch.isnan(output)] = sensor.cfg.max_distance
    output[torch.isinf(output)] = sensor.cfg.max_distance

    # normalize the data
    # output = torch.clip(output, 0.0, sensor.cfg.max_distance) / sensor.cfg.max_distance
    return output


def height_scan_inf_filtered(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.ray_hits_w[..., 2] + offset - sensor.data.pos_w[:, 2].unsqueeze(1)
    # assign max distance to inf values
    height[torch.isinf(height)] = sensor.cfg.max_distance
    height[torch.isnan(height)] = sensor.cfg.max_distance

    return height


def height_scan_square(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor."""
    # call regular height scanner function
    height = height_scan_inf_filtered(env, sensor_cfg, offset=offset)
    shape = shape if shape is not None else [int(math.sqrt(height.shape[1])), int(math.sqrt(height.shape[1]))]
    # unflatten the height scan to make use of spatial information
    height_square = torch.unflatten(height, 1, (shape[0], shape[1]))
    # the height scan is mirrored as the pattern is created from neg to pos whereas in the robotics frame, the left of
    # the robot is positive and the right is negative
    height_square = torch.flip(height_square, dims=[1])
    # unqueeze to make compatible with convolutional layers
    return height_square.unsqueeze(1)


def height_scan_door_recognition(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    door_height_thres: float = 1.25,
    offset: float = 0.5,
    return_height: bool = True,
) -> torch.Tensor | None:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for doors in the scene."""

    # extract the used quantities (to enable type-hinting)
    sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, MultiMeshRayCaster), "The sensor must be a MultiMeshRayCaster."

    # get the sensor hit points
    ray_origins = sensor.data.ray_hits_w.clone()

    # we raycast one more time shortly above the ground up and down, if the up raycast hits and is lower than the
    # initial raycast, a potential door is detected
    ray_origins[..., 2] = 0.5
    ray_directions = torch.zeros_like(ray_origins)
    ray_directions[..., 2] = -1.0

    hit_point_down = raycast_dynamic_meshes(
        ray_origins,
        ray_directions,
        mesh_ids_wp=sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
        max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w if sensor.cfg.track_mesh_transforms else None,
        mesh_orientations_w=sensor._mesh_orientations_w if sensor.cfg.track_mesh_transforms else None,
    )[0]

    ray_directions[..., 2] = 1.0

    hit_point_up = raycast_dynamic_meshes(
        ray_origins,
        ray_directions,
        mesh_ids_wp=sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
        max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w if sensor.cfg.track_mesh_transforms else None,
        mesh_orientations_w=sensor._mesh_orientations_w if sensor.cfg.track_mesh_transforms else None,
    )[0]

    lower_height = (
        (hit_point_up[..., 2] < (sensor.data.ray_hits_w[..., 2] - 1e-3))
        & torch.isfinite(hit_point_up[..., 2])
        & ((hit_point_up[..., 2] - hit_point_down[..., 2]) > door_height_thres)
        & torch.isfinite(hit_point_down[..., 2])
    )

    # overwrite the data
    sensor.data.ray_hits_w[lower_height] = hit_point_down[lower_height]

    # debug
    if False:
        env_render_steps = 1000

        # provided height scan
        positions = sensor.data.ray_hits_w.clone()
        # flatten positions
        positions = positions.view(-1, 3)

        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            draw_interface.draw_points(
                positions.tolist(),
                [(1.0, 0.5, 0, 1)] * positions.shape[0],
                [5] * positions.shape[0],
            )

            sim = SimulationContext.instance()
            for _ in range(env_render_steps):
                sim.render()

            # clear the drawn points and lines
            draw_interface.clear_points()
            draw_interface.clear_lines()

        except ImportError:
            print("[WARNING] Cannot visualize occluded height scan in headless mode.")

    if return_height:
        # call regular height scanner function
        return height_scan_square(env, sensor_cfg, shape, offset)
    else:
        return None


@configclass
class HeightScanOcculusionModifierCfg:
    """Configuration for the HeightScanOcculusionModifier."""

    height_scan_func: callable = MISSING
    """The height scan function to modify."""

    sensor_cfg: SceneEntityCfg = MISSING
    """The sensor configuration."""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """The asset configuration."""

    env_ratio: float | None = None
    """The ratio of environments to apply the occlusion to."""

    sensor_offsets: list[list[float]] | list[float] | None = None
    """The sensor offset to account for the sensor's position."""

    offset_threshold: float = 0.5  # 0.01
    """The distance threshold to consider a point as occluded."""

    def __post_init__(self):
        assert (
            self.env_ratio is None or 0.0 <= self.env_ratio <= 1.0
        ), "The environment ratio must be between 0.0 and 1.0."


class HeightScanOcculusionModifier:
    """Modify height scan to account for occulsions in the terrain that cannot be observed by the sensor."""

    def __init__(self, cfg: HeightScanOcculusionModifierCfg):
        self.cfg = cfg

    def _setup(self, env: ManagerBasedRLEnv):
        # extract the used quantities (to enable type-hinting)
        self._sensor: MultiMeshRayCaster = env.scene.sensors[self.cfg.sensor_cfg.name]
        self._asset: Articulation = env.scene[self.cfg.asset_cfg.name]
        assert isinstance(self._sensor, MultiMeshRayCaster), "The sensor must be a MultiMeshRayCaster."
        # account for the sensor offset
        if self.cfg.sensor_offsets is not None:
            if isinstance(self.cfg.sensor_offsets[0], list):
                self._sensor_offset_tensor = (
                    torch.tensor(self.cfg.sensor_offsets, device=self._asset.device)
                    .unsqueeze(1)
                    .repeat(1, env.num_envs, 1)
                )
            else:
                self._sensor_offset_tensor = torch.tensor(
                    [[self.cfg.sensor_offsets]], device=self._asset.device
                ).repeat(1, env.num_envs, 1)
        else:
            self._sensor_offset_tensor = None
        # get the sensors where occlusion should be applied
        if self.cfg.env_ratio is not None:
            self._env_ids = torch.randperm(env.num_envs, device=env.device)[: int(self.cfg.env_ratio * env.num_envs)]
        else:
            self._env_ids = slice(None)

    def _get_occuled_points(self, robot_position: torch.Tensor) -> torch.Tensor:
        robot_position = robot_position[:, None, :].repeat(1, self._sensor.data.ray_hits_w.shape[1], 1)
        ray_directions = self._sensor.data.ray_hits_w - robot_position

        # NOTE: ray directions can never be inf or nan, otherwise the raycasting takes forever
        ray_directions[torch.isinf(ray_directions)] = 0.0
        ray_directions[torch.isnan(ray_directions)] = 0.0

        # raycast from the robot to intended hit positions
        ray_hits_w = raycast_dynamic_meshes(
            robot_position,
            ray_directions,
            mesh_ids_wp=self._sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self._sensor.cfg.max_distance,
            mesh_positions_w=self._sensor._mesh_positions_w if self._sensor.cfg.track_mesh_transforms else None,
            mesh_orientations_w=self._sensor._mesh_orientations_w if self._sensor.cfg.track_mesh_transforms else None,
        )[0]

        # get not visible parts of the height-scan
        unseen = torch.norm(ray_hits_w - self._sensor.data.ray_hits_w, dim=2) > self.cfg.offset_threshold

        return unseen

    def __call__(self, env: ManagerBasedRLEnv, *args, **kwargs) -> torch.Tensor:
        """Modify the height scan to account for occulsions in the terrain that cannot be observed by the sensor."""

        # setup the modifier
        if not hasattr(self, "_sensor"):
            self._setup(env)

        # account for the sensor offset
        if self._sensor_offset_tensor is not None:
            unseen = torch.zeros(
                (self._sensor_offset_tensor.shape[0], *self._sensor.data.ray_hits_w.shape[:-1]),
                device=self._asset.device,
                dtype=torch.bool,
            )
            for offset_idx in range(self._sensor_offset_tensor.shape[0]):
                robot_position = self._asset.data.root_pos_w + math_utils.quat_rotate(
                    self._asset.data.root_quat_w, self._sensor_offset_tensor[offset_idx]
                )
                unseen[offset_idx] = self._get_occuled_points(robot_position)
            unseen = torch.all(unseen, dim=0)
        else:
            robot_position = self._asset.data.root_pos_w

        # overwrite the data
        unseen[self._env_ids] = False
        if torch.any(unseen):
            unseen_points = self._sensor.data.ray_hits_w[unseen]
            unseen_points[..., 2] = self._sensor.cfg.max_distance
            self._sensor.data.ray_hits_w[unseen] = unseen_points

        # return the modified height scan
        return self.cfg.height_scan_func(env, *args, **kwargs)

    def __name__(self):
        return "HeightScanOcculusionModifier"


class HeightScanOcculusionDoorRecognitionModifier(HeightScanOcculusionModifier):

    def __init__(self, cfg: HeightScanOcculusionModifierCfg):
        super().__init__(cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        shape: list[int] | None = None,
        door_height_thres: float = 1.25,
        offset: float = 0.5,
    ):
        height_scan_door_recognition(env, sensor_cfg, door_height_thres=door_height_thres, return_height=False)
        return super().__call__(env, sensor_cfg=sensor_cfg, shape=shape, offset=offset)

    def __name__(self):
        return "HeightScanOcculusionDoorRecognitionModifier"


def height_scan_square_exp_occlu(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for occulsions of the terrain."""

    # extract the used quantities (to enable type-hinting)
    sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    assert isinstance(sensor, MultiMeshRayCaster), "The sensor must be a MultiMeshRayCaster."

    # get the sensor hit points
    ray_hits = sensor.data.ray_hits_w.clone()
    # account for the sensor offset
    robot_position = asset.data.root_pos_w + math_utils.quat_rotate(
        asset.data.root_quat_w, torch.tensor([[0.4, 0.0, 0.0]], device=asset.device).repeat(env.num_envs, 1)
    )
    robot_position = robot_position[:, None, :].repeat(1, ray_hits.shape[1], 1)
    ray_directions = ray_hits - robot_position

    # NOTE: ray directions can never be inf or nan, otherwise the raycasting takes forever
    ray_directions[torch.isinf(ray_directions)] = 0.0
    ray_directions[torch.isnan(ray_directions)] = 0.0

    # raycast from the robot to intended hit positions
    ray_hits_w = raycast_dynamic_meshes(
        robot_position,
        ray_directions,
        mesh_ids_wp=sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
        max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w if sensor.cfg.track_mesh_transforms else None,
        mesh_orientations_w=sensor._mesh_orientations_w if sensor.cfg.track_mesh_transforms else None,
    )[0]

    # get not visible parts of the height-scan
    unseen = torch.norm(ray_hits_w - ray_hits, dim=2) > 0.01

    # overwrite the data
    if torch.any(unseen):
        unseen_points = sensor.data.ray_hits_w[unseen]
        unseen_points[..., 2] = sensor.cfg.max_distance
        sensor.data.ray_hits_w[unseen] = unseen_points

    # debug
    if False:
        env_render_steps = 1000

        # provided height scan
        positions = sensor.data.ray_hits_w.clone()
        # flatten positions
        positions = positions.view(-1, 3)

        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            draw_interface.draw_points(
                positions.tolist(),
                [(1.0, 0.5, 0, 1)] * positions.shape[0],
                [5] * positions.shape[0],
            )

            sim = SimulationContext.instance()
            for _ in range(env_render_steps):
                sim.render()

            # clear the drawn points and lines
            draw_interface.clear_points()
            draw_interface.clear_lines()

        except ImportError:
            print("[WARNING] Cannot visualize occluded height scan in headless mode.")

    # run regular height scan
    return height_scan_square(env, sensor_cfg, shape, offset)


def height_scan_square_exp_occlu_with_door_recognition(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    door_height_thres: float = 1.25,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for occulsions of the terrain and doors in the scene.
    """

    height_scan_door_recognition(
        env,
        sensor_cfg,
        shape,
        door_height_thres=door_height_thres,
        offset=offset,
        return_height=False,
    )
    return height_scan_square_exp_occlu(env, asset_cfg, sensor_cfg, shape, offset)


"""
Collision
"""


def base_collision(
    env: ManagerBasedRLEnv, threshold: float = 1.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    normed_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1).flatten(start_dim=1)
    return (torch.max(normed_force, dim=1)[0] > threshold).unsqueeze(-1)


"""
Actions.
"""


def last_low_level_action(
    env: ManagerBasedRLEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The last low-level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions[:, asset_cfg.joint_ids]


def second_last_low_level_action(
    env: ManagerBasedRLEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The second to last low level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.prev_low_level_actions[:, asset_cfg.joint_ids]


"""
Commands.
"""


def vel_commands(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The velocity command generated by the planner and given as input to the step function"""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.processed_actions


def goal_command_w_se2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command_term: GoalCommand = env.command_manager._terms[command_name]
    goal = command_term.pos_command_w.clone()
    goal[:, 2] = 0.0
    return goal


def goal_command_w_se2_mixed(env: ManagerBasedRLEnv, command_name: str, subterm_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    mixed_command_term: MixedCommand = env.command_manager._terms[command_name]
    command_term: GoalCommand = mixed_command_term.get_subterm(subterm_name)
    goal = command_term.pos_command_w.clone()
    goal[:, 2] = 0.0
    return goal


"""
Energy consumption
"""


def energy_consumption(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), energy_scale_factor: float = 0.001
) -> torch.Tensor:
    """The energy consumption of the asset. Computed as the sum of the squared applied torques."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.applied_torque**2).sum(dim=-1).unsqueeze(-1) * energy_scale_factor
