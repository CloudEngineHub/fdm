# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.terrains as terrain_gen

import nav_tasks.terrains as fdm_terrain_gen

from .terrains import RslStairsCfg

###
# Baseline 2D Environment
###

BASELINE_2D_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(5e-3, 1e-2),
            noise_step=1e-2,
            border_width=0.25,
            vertical_scale=1e-3,
            proportion=0.5,
        ),
        "flat_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(2.5, 2.5), num_objects=(5, 7)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(2.5, 2.5), num_objects=(5, 7)
            ),
        ),
        # "rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25, proportion=0.5,
        # ),
        # "rough_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
        #     box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
        #         width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(2.5, 2.5), num_objects=(5, 7)
        #     ),
        #     cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
        #         radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(2.5, 2.5), num_objects=(5, 7)
        #     ),
        #     rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
        #         noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25
        #     ),
        # ),
        "all_wall": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.4, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.5,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            no_free_space_front=True,
            all_wall=True,
            max_height=2.5,
        ),
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 2.0],
            proportion=0.5,
        ),
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 1.0],
            proportion=0.5,
        ),
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 2.0],
            proportion=0.5,
        ),
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 1.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.5,
        ),
        "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.25, 0.5],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.5,
        ),
        "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 2.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.5,
            wall_width=0.2,
        ),
        # "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
        #     object_type="box",
        #     dim_range=[0.5, 1.5],
        #     height_range=[2.5, 2.5],
        #     position_pattern=fdm_terrain_gen.extended_cross_object_pattern,
        # ),
        # "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
        #     object_type="cylinder",
        #     dim_range=[0.25, 0.75],
        #     height_range=[2.5, 2.5],
        #     position_pattern=fdm_terrain_gen.extended_cross_object_pattern,
        # ),
        # "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
        #     object_type="wall",
        #     dim_range=[1.0, 2.5],
        #     height_range=[2.5, 2.5],
        #     position_pattern=fdm_terrain_gen.extended_cross_object_pattern,
        # ),
    },
    border_height=2.5,
)


##
# Terrain Generator
##

FDM_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "outdoor": fdm_terrain_gen.MeshPillarTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.3, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(2.5, 2.5), num_objects=(3, 7)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(2.5, 2.5), num_objects=(3, 7)
            ),
        ),
    },
)


PILLAR_TERRAIN_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=9,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pillar_eval": fdm_terrain_gen.MeshPillarTerrainDeterministicCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainDeterministicCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(2, 3)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainDeterministicCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(2, 3)
            ),
        ),
        "stairs_eval": fdm_terrain_gen.MeshQuadPyramidStairsCfg(
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=0.5,
            holes=False,
        ),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.75,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
            holes=False,
        ),
        "rampIncline_stairs_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(5, 45),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "ramp_stairsIncline_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.05, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "ramp_stairs_wall": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.05, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            random_wall_probability=1.0,
            max_height=1.0,
        ),
        "all_wall": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.05, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.5,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            all_wall=True,
            max_height=1.0,
        ),
        "box": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.5, 2.5),
            platform_width=3.0,
        ),
        "rsl_stairs": RslStairsCfg(),
    },
    border_height=2.0,
)


FDM_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "outdoor": fdm_terrain_gen.MeshPillarTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(7, 7)
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25
            ),
        ),
    },
)


FDM_EXTEROCEPTIVE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_height=3.0,
    sub_terrains={
        "outdoor": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.4,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(7, 7)
            ),
        ),
        "outdoor_rough": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.3,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(7, 7)
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25
            ),
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.75,
            grid_height_range=(0.05, 0.25),
            platform_width=0.2,
            holes=False,
        ),
    },
)


FDM_EVAL_EXTEROCEPTIVE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=1.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(5e-3, 1e-2), noise_step=1e-2, border_width=0.25, vertical_scale=1e-3
        ),
        "flat_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.25,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(7, 7)
            ),
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "rough_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.25,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(7, 7)
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25
            ),
        ),
    },
)

MAZE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=1.0,
    border_height=3.0,
    num_cols=8,
    num_rows=6,
    use_cache=False,
    sub_terrains={
        "maze": fdm_terrain_gen.RandomMazeTerrainCfg(
            proportion=1.0,
            resolution=1.25,
            maze_height=3.0,
            step_height_range=(0.15, 0.25),
            step_width_range=(0.25, 0.35),
            num_stairs=5,
        ),
    },
)
MAZE_MERGE_TERRAIN_CFG = MAZE_TERRAIN_CFG.replace(num_cols=3, num_rows=6)

BASELINE_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "semi_flat": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=6.75,
            grid_height_range=(0.05, 0.05),
            platform_width=12.0,
            holes=False,
        ),
    },
    border_height=2.0,
)

###
# FDM ACCURACY EVALUATION TERRAINS
###


PILLAR_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # "deterministic": fdm_terrain_gen.MeshPillarTerrainDeterministicCfg(
        #     box_objects=fdm_terrain_gen.MeshPillarTerrainDeterministicCfg.BoxCfg(
        #         width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(2, 3)
        #     ),
        #     cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainDeterministicCfg.CylinderCfg(
        #         radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(2, 3)
        #     ),
        # ),
        "random": fdm_terrain_gen.MeshPillarTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(1.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(1.5, 3), num_objects=(7, 7)
            ),
            proportion=1.0,
        ),
        # "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
        #     object_type="box",
        #     dim_range=[0.5, 1.5],
        #     height_range=[1.0, 2.0],
        #     position_pattern=fdm_terrain_gen.extended_cross_object_pattern,
        #     proportion=0.25,
        # ),
        # "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
        #     object_type="cylinder",
        #     dim_range=[0.25, 0.75],
        #     height_range=[1.0, 2.0],
        #     position_pattern=fdm_terrain_gen.extended_cross_object_pattern,
        #     proportion=0.25,
        # ),
        # "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
        #     object_type="wall",
        #     dim_range=[1.0, 2.5],
        #     height_range=[1.0, 2.0],
        #     position_pattern=fdm_terrain_gen.extended_cross_object_pattern,
        #     proportion=0.25,
        # ),
    },
    border_height=3.0,
)


# NOTE: not used anymore
GRID_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.75,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
            holes=False,
        ),
    },
    border_height=3.0,
)


STAIRS_WALL_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=20,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "all_wall_step": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.15, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.5,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            no_free_space_front=True,
            random_wall_probability=0.6,
            all_wall=False,
            max_height=1.25,
            proportion=1.0,
        ),
        "all_wall_ramp": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(20, 45),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.5,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            no_free_space_front=True,
            random_wall_probability=0.6,
            all_wall=False,
            max_height=1.25,
            proportion=1.0,
        ),
        "box": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.5, 1.5),
            platform_width=3.0,
            proportion=0.25,
        ),
        "stairs_eval": fdm_terrain_gen.MeshQuadPyramidStairsCfg(
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=0.5,
            holes=False,
            proportion=0.25,
        ),
    },
    border_height=3.0,
)


# NOTE: not used anymore
STAIRS_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs_eval": fdm_terrain_gen.MeshQuadPyramidStairsCfg(
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=0.5,
            holes=False,
        ),
    },
    border_height=3.0,
)


STAIRS_RAMP_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=20,
    num_cols=15,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "rampIncline_stairs_freeFront": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(20, 45),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=2.5,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
            random_wall_probability=0.0,
            free_space_front=False,
            no_free_space_front=False,
        ),
        "ramp_stairsIncline_freeFront": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.15, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=2.5,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
            random_wall_probability=0.0,
            free_space_front=False,
            no_free_space_front=False,
        ),
        "rampIncline_stairs_freeBack": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(20, 45),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=2.5,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
            random_wall_probability=0.0,
            free_space_front=False,
            no_free_space_front=True,
        ),
        "ramp_stairsIncline_freeBack": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.10, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=2.5,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
            random_wall_probability=0.0,
            free_space_front=False,
            no_free_space_front=True,
        ),
        "ramp_stairs_wall": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.05, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=2.5,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            random_wall_probability=0.0,
            max_height=1.0,
        ),
    },
    border_height=3.0,
)


# NOTE: not used anymore
STAIRS_RAMP_LARGE_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=4,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "ramp_incline_front": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(25, 45),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.75,
            free_space_front=False,
            width_randomization=2.0,
            random_stairs_ramp_position_flipping=True,
            max_height=0.75,
        ),
        "stairs_incline_front": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.25, 0.4),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.75,
            free_space_front=False,
            width_randomization=2.0,
            random_stairs_ramp_position_flipping=True,
            max_height=0.75,
        ),
    },
    border_height=3.0,
)

###
# FDM PLANNING EVAL TERRAINS
###

PLANNER_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=2,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "ramp_incline_back": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(30, 45),
            step_width=0.3,
            platform_width=2.0,
            border_width=2.0,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "stairs_incline_back": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.2, 0.4),
            step_width=0.3,
            platform_width=2.0,
            border_width=2.0,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "rampIncline_stairs_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            proportion=2.0,
            modify_ramp_slope=True,
            ramp_slope_range=(30, 45),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "ramp_stairsIncline_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            proportion=2.0,
            modify_step_height=True,
            step_height_range=(0.15, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 2.0],
        ),
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 1.0],
        ),
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 2.0],
        ),
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 1.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
        ),
        "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.25, 0.5],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
        ),
        "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 2.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
        ),
    },
    border_height=3.0,
)

PLANNER_EVAL_2D_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=12,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pillar": fdm_terrain_gen.MeshPillarPlannerTestTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.4, 0.7), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(1.5, 2), num_objects=(2, 3)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(1.5, 2), num_objects=(2, 3)
            ),
            proportion=1.0,
            platform_width=2.0,
        ),
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 2.0],
            proportion=0.5,
        ),
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 1.0],
            proportion=0.5,
        ),
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 2.0],
            proportion=0.5,
        ),
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 1.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.5,
        ),
        "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.25, 0.5],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.5,
        ),
        "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 2.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.5,
            wall_width=0.2,
        ),
    },
    border_height=0.0,
)


PLANNER_EVAL_3D_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=12,
    num_cols=8,  # 10
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "ramp_incline_back": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(15, 45),
            step_width=0.3,
            platform_width=2.0,
            border_width=2.0,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "stairs_incline_back": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.15, 0.4),
            step_width=0.3,
            platform_width=2.0,
            border_width=2.0,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "rampIncline_stairs_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            proportion=2.0,
            modify_ramp_slope=True,
            ramp_slope_range=(15, 45),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "ramp_stairsIncline_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            proportion=2.0,
            modify_step_height=True,
            step_height_range=(0.15, 0.4),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
    },
    border_height=3.0,
)


###
# PAPER PLOT TERRAINS
###

PAPER_FIGURE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=0.0,
    num_rows=1,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=6.75,
            grid_height_range=(0.05, 0.05),
            platform_width=12.0,
            holes=False,
        ),
        "pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.4, 0.4), max_yx_angle=(0, 10), height=(1.5, 3), num_objects=(5, 5)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(1.5, 3), num_objects=(7, 7)
            ),
            platform_width=2.5,
        ),
        "ramp_stairsIncline_eval": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.3, 0.3),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=2.5,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
            random_wall_probability=0.0,
        ),
        "wall": fdm_terrain_gen.StairsRampEvalTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.2, 0.2),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            border_width=0.25,
            width_randomization=1.5,
            random_stairs_ramp_position_flipping=True,
            free_space_front=False,
            # all_wall=True,
            random_wall_probability=0.8,
            max_height=1.0,
        ),
    },
    border_height=0.2,
)


PAPER_PLATFORM_FIGURE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(18.0, 18.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs_ramp": fdm_terrain_gen.StairsRampUpDownTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.15, 0.15),
            step_width=0.3,
            platform_width=3.0,
            center_platform_width=1.0,
            border_width=0.0,
            width_randomization=2.0,
            random_stairs_ramp_position_flipping=False,
            max_height=1.0,
            random_wall_probability=0.0,
            no_free_space_front=False,
        ),
    },
    border_height=0.2,
)


PAPER_PLANNER_FIGURE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(7.0, 7.0),
    border_width=1.0,
    num_rows=1,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pillar": fdm_terrain_gen.MeshPillarPlannerTestTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.3, 0.5), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(1.5, 2), num_objects=(3, 3)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.2, 0.3), max_yx_angle=(0, 5), height=(1.5, 2), num_objects=(3, 3)
            ),
            platform_width=2.5,
            border_width=0.5,
        ),
        "ramp_incline_back": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(35, 35),
            step_width=0.3,
            platform_width=1.3,
            border_width=0.5,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
        ),
        "stairs_incline_back": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.25, 0.25),
            step_width=0.3,
            platform_width=1.3,
            border_width=0.5,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=False,
            max_height=1.0,
        ),
        "stairs_wall": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_step_height=True,
            step_height_range=(0.25, 0.25),
            step_width=0.3,
            platform_width=1.3,
            border_width=0.5,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=False,
            max_height=1.0,
            random_wall_probability=1.0,
        ),
        "ramp_wall": fdm_terrain_gen.StairsRampTerrainCfg(
            modify_ramp_slope=True,
            ramp_slope_range=(35, 35),
            step_width=0.3,
            platform_width=1.3,
            border_width=0.5,
            width_randomization=1.0,
            random_stairs_ramp_position_flipping=True,
            max_height=1.0,
            random_wall_probability=1.0,
        ),
    },
    border_height=0.0,
)
