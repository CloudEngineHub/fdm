

from dataclasses import MISSING

from omni.isaac.lab.terrains import SubTerrainBaseCfg
from omni.isaac.lab.terrains.height_field import HfRandomUniformTerrainCfg
from omni.isaac.lab.utils import configclass

from .pillar_terrain import pillar_terrain, pillar_terrain_eval


@configclass
class MeshPillarTerrainCfg(SubTerrainBaseCfg):
    @configclass
    class CylinderCfg:
        """Configuration for repeated cylinder."""

        radius: tuple[float, float] = MISSING
        """The radius of the pyramids (in m). First value start of curriculum, second value end."""
        max_yx_angle: tuple[float, float] = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0. First value start of curriculum, second value end."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""
        num_objects: tuple[int, int] = MISSING
        """The number of objects to add to the terrain. First value start of curriculum, second value end."""
        height: tuple[float, float] = MISSING
        """The height (along z) of the object (in m). First value start of curriculum, second value end."""
        object_type: str = "cylinder"
        """The type of object to generate.

        The type can be a string or a callable. If it is a string, the function will look for a function called
        ``make_{object_type}`` in the current module scope. If it is a callable, the function will
        use the callable to generate the object.
        """

    @configclass
    class BoxCfg:
        """Configuration for repeated boxes."""

        width: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m). First value start of curriculum, second value end."""
        length: tuple[float, float] = MISSING
        """The length (along y) of the box (in m). First value start of curriculum, second value end."""
        max_yx_angle: tuple[float, float] = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0. First value start of curriculum, second value end."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""
        num_objects: tuple[int, int] = MISSING
        """The number of objects to add to the terrain. First value start of curriculum, second value end."""
        height: tuple[float, float] = MISSING
        """The height (along z) of the object (in m). First value start of curriculum, second value end."""
        object_type: str = "box"
        """The type of object to generate.

        The type can be a string or a callable. If it is a string, the function will look for a function called
        ``make_{object_type}`` in the current module scope. If it is a callable, the function will
        use the callable to generate the object.
        """

    function = pillar_terrain

    rough_terrain: HfRandomUniformTerrainCfg = None
    """The configuration for the rough terrain. If None, the terrain will be flat. Defaults to None."""
    box_objects: BoxCfg = MISSING
    """add boxes to the terrain"""
    cylinder_cfg: CylinderCfg = MISSING
    """add cylinders to the terrain"""

    max_height_noise: float = 0.0
    """The maximum amount of noise to add to the height of the objects (in m). Defaults to 0.0."""
    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshPillarTerrainEvalCfg(MeshPillarTerrainCfg):

    function = pillar_terrain_eval
    """The function to call to evaluate the terrain."""

    max_obstacle_distance: float = 4.0

    platform_width: float = 2.0
