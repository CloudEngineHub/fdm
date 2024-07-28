

"""Forward Dynamics Model specific randomization utilities."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_from_euler_xyz, sample_uniform
from nav_collectors.terrain_analysis import TerrainAnalysis, TerrainAnalysisCfg
from nav_tasks.mdp import GoalCommand

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def regular_rigid_body_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Assign the physics materials on all geometries of the asset with a linear pattern.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)

    # create material properties for each env, these are linearly interpolated between min and max value for the static
    # friction coefficient
    material_envs = torch.zeros(num_envs, 3)
    material_envs[:, 0] = torch.arange(
        static_friction_range[0],
        static_friction_range[1],
        step=(static_friction_range[1] - static_friction_range[0]) / num_envs,
    )
    material_envs[:, 1] = material_envs[:, 0].clone()
    bodies_per_env = asset.body_physx_view.count // num_envs  # - number of bodies per spawned asset
    material_envs = (
        material_envs.unsqueeze(1).unsqueeze(-1).repeat(1, bodies_per_env, 1, asset.body_physx_view.max_shapes)
    )
    material_envs = material_envs.view(-1, 3, asset.body_physx_view.max_shapes)
    # resolve the global body indices from the env_ids and the env_body_ids
    indices = torch.tensor(asset_cfg.body_ids, dtype=torch.int).repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    # set the material properties into the physics simulation
    # TODO: Need to use CPU tensors for now. Check if this changes in the new release
    asset.body_physx_view.set_material_properties(material_envs, indices.view(-1))

    env.extras = {"friction": material_envs[indices.view(-1)]}


def reset_root_state_center(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to zero velocity, forward orientation and a linearly spaced position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions (small perturbation to avoid spawning issues)
    torch.manual_seed(0)
    perturbation = torch.hstack([
        sample_uniform(-0.2, 0.2, (len(env_ids), 2), device=asset.device),
        torch.zeros((len(env_ids), 1), device=asset.device),
    ])
    perturbation[:, 2] = 0.1
    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + perturbation
    # orientations
    num_env_origins = torch.unique(env.scene.env_origins, dim=0).shape[0]
    assets_per_origin = math.ceil(env.num_envs / num_env_origins)
    yaw_orientation = torch.linspace(-torch.pi, torch.pi, assets_per_origin + 1, device=env.device)[:-1]
    yaw_samples = yaw_orientation[env_ids % assets_per_origin]
    orientations = quat_from_euler_xyz(torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples)

    # velocities
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_regular(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to zero velocity, forward orientation and a linearly spaced position."""
    assert env.num_envs % (env.scene.terrain.cfg.terrain_generator.num_rows - 1) == 0, (
        "Number of environments must be divisible by number of rows - 1 in the terrain generator. "
        f"Number of environments: {env.num_envs}, number of rows: {env.scene.terrain.cfg.terrain_generator.num_rows}"
    )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions
    # FIXME: what if terrain is not generated?
    raise NotImplementedError("Need to update this function.")
    horizontal_size = env.scene.terrain.cfg.terrain_generator.size[0] * env.scene.terrain.cfg.terrain_generator.num_cols
    vertical_size = env.scene.terrain.cfg.terrain_generator.size[1] * env.scene.terrain.cfg.terrain_generator.num_rows
    positions = root_states[:, 0:3].clone()
    positions[:, 1] = torch.arange(
        1 - (horizontal_size / 2), horizontal_size / 2, step=(horizontal_size - 1) / env.num_envs * 2, device=env.device
    ).repeat(2)[env_ids]
    positions[:, 0] = torch.arange(
        -(vertical_size / 2),
        vertical_size / 2,
        step=vertical_size / env.scene.terrain.cfg.terrain_generator.num_rows,
        device=env.device,
    )[1:].repeat(int(env.num_envs / (env.scene.terrain.cfg.terrain_generator.num_rows - 1)))[env_ids]
    # orientations
    orientations = torch.zeros_like(root_states[:, 3:7])
    orientations[:, 0] = 1.0
    # velocities
    velocities = torch.zeros_like(root_states[:, 7:13])

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


###
# Terrain Analysis based Reset
###


class TerrainAnalysisRootReset:

    def __init__(self, cfg: TerrainAnalysisCfg):
        self.cfg = cfg

        self.analyser = None

    def _run_analysis(self, env: ManagerBasedRLEnv):
        """Run the terrain analysis to compute the root state reset."""
        self.analyser = TerrainAnalysis(self.cfg, env.scene)
        print("[INFO] Running terrain analysis")
        self.analyser.analyse()
        self.analyser.points = self.analyser.points.to(env.device)
        print("[INFO] Terrain analysis completed")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        yaw_range: tuple[float, float],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        """The function samples the new start positions in the terrain with random initial velocities."""

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

        # positions
        if self.analyser is None:
            self._run_analysis(env)
        positions = self.analyser.points[torch.randperm(self.analyser.points.shape[0])[: len(env_ids)]]

        # yaw range
        yaw_samples = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=asset.device)
        orientations = quat_from_euler_xyz(
            torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples
        ).squeeze(1)

        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    def __name__(self):
        # return the name of the function for logging purposes
        return "TerrainAnalysisRootReset"


###
# Planner Reset
###


def reset_robot_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to the spawn state defined by the command generator.

    Args:
        env: The environment object.
        env_ids: The environment ids to reset.
        asset_cfg: The asset configuration to reset. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms["command"]

    # positions - based on given start points (command generator)
    positions = goal_cmd_geneator.pos_spawn_w[env_ids]
    # orientations - randomly sampled in the range of the -pi to +pi
    rand_samples = sample_uniform(-torch.pi, torch.pi, len(env_ids), device=asset.device)
    orientations = quat_from_euler_xyz(
        torch.zeros(len(env_ids), device=asset.device), torch.zeros(len(env_ids), device=asset.device), rand_samples
    )
    # velocities - zero
    velocities = asset.data.default_root_state[env_ids, 7:13]
    # set into the physics simulation
    asset.write_root_state_to_sim(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)
    # obtain default joint positions
    default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # set into the physics simulation
    asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
