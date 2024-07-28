


from omni.isaac.lab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg
from omni.isaac.lab.utils import configclass

from .stairs_terrain import pyramid_stairs_eval_terrain


@configclass
class MeshStairsEvalCfg(MeshPyramidStairsTerrainCfg):

    function = pyramid_stairs_eval_terrain
    """The function to call to evaluate the terrain."""
