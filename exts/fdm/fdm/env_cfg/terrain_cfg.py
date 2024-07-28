

import omni.isaac.lab.terrains as terrain_gen

import fdm.terrains as fdm_terrain_gen

##
# Terrain Generator
##

FDM_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
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
        ),
    },
)


PILLAR_TERRAIN_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=5,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pillar_eval": fdm_terrain_gen.MeshPillarTerrainEvalCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainEvalCfg.BoxCfg(
                width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(2, 3)
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainEvalCfg.CylinderCfg(
                radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(2, 3)
            ),
        ),
        "stairs_eval": fdm_terrain_gen.MeshStairsEvalCfg(
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
    },
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
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
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
