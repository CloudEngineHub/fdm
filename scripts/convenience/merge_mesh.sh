#!/usr/bin/env bash

ISAACLAB_HOME=/home/pascal/orbit/IsaacLab

${ISAACLAB_HOME}/isaaclab.sh -p fdm/isaac-nav-suite/scripts/tools/mesh_merger.py --env_list \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_0/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_1/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_2/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_3/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_4/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_5/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_6/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_7/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_8/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_9/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_10/mesh.obj" \
    "/home/pascal/nav_projects/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_11/mesh.obj" \
    --generator_list "PLANNER_TRAIN_CFG" "FDM_EXTEROCEPTIVE_TERRAINS_CFG" "MAZE_MERGE_TERRAIN_CFG" # "BASELINE_FLAT_TERRAIN_CFG"


# all terrains
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_0/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_1/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_2/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_3/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_4/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_5/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_6/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_7/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_8/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_9/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_10/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all/mesh_11/mesh.obj",

# all terrains with walls (emptier)
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_0/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_1/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_2/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_3/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_4/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_5/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_6/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_7/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_8/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_9/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_10/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_all_wall_emptier/mesh_11/mesh.obj",

# stairs
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_stairs/mesh_0/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_stairs/mesh_1/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_stairs/mesh_2/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_stairs/mesh_3/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_stairs/mesh_4/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_stairs/mesh_5/mesh.obj",

# ramp and platform
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_ramp_platform/mesh_0/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_ramp_platform/mesh_1/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_ramp_platform/mesh_2/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_ramp_platform/mesh_3/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_ramp_platform/mesh_4/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_ramp_platform/mesh_5/mesh.obj",

# perlin and stepping stones
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_0/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_1/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_2/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_3/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_4/mesh.obj",
# "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_5/mesh.obj",
