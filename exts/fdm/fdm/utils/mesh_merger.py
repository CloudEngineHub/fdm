

"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--env_list",
    type=list,
    default=[
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
        "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_0/mesh.obj",
        "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_1/mesh.obj",
        "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_2/mesh.obj",
        "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_3/mesh.obj",
        "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_4/mesh.obj",
        "/home/pascal/memory_iplanner/terrain_generator_new/results/generated_terrain_perlin_stepping_stones/mesh_5/mesh.obj",
    ],
    help="List of environments that should be combined.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import trimesh

import carb
import omni.ext
import omni.isaac.core.utils.prims as prim_utils
import omni.kit.commands
import omni.kit.utils
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg
from omni.isaac.lab.terrains.trimesh.utils import make_border


def _merge_mesh(curr_prim: Usd.Prim):
    stage = omni.usd.get_context().get_stage()
    prim_transform = omni.usd.get_world_transform_matrix(curr_prim, Usd.TimeCode.Default())
    count = 0
    meshes = []
    for child_prim in Usd.PrimRange(curr_prim, Usd.TraverseInstanceProxies()):
        imageable = UsdGeom.Imageable(child_prim)
        visible = imageable.ComputeVisibility(Usd.TimeCode.Default())
        if (
            child_prim.IsA(UsdGeom.Mesh)
            and visible != UsdGeom.Tokens.invisible
            and imageable.GetPurposeAttr().Get() in ["default", "render"]
        ):
            carb.log_warn(child_prim.GetName())
            usdMesh = UsdGeom.Mesh(child_prim)
            mesh = {"points": usdMesh.GetPointsAttr().Get()}
            world_mtx = omni.usd.get_world_transform_matrix(child_prim, Usd.TimeCode.Default())
            # if self.parent_xform.get_value_as_bool():
            #     world_mtx = prim_transform * world_mtx * prim_transform.GetInverse()
            # else:
            world_mtx = world_mtx * prim_transform.GetInverse()
            world_rot = world_mtx.ExtractRotation()
            # print(world_mtx)
            mesh["points"][:] = [world_mtx.TransformAffine(x) for x in mesh["points"]]
            mesh["normals"] = usdMesh.GetNormalsAttr().Get()
            mesh["attr_normals"] = usdMesh.GetPrim().GetAttribute("primvars:normals").Get()
            mesh["attr_normals_indices"] = usdMesh.GetPrim().GetAttribute("primvars:normals:indices").Get()
            if not mesh["attr_normals"]:
                mesh["attr_normals"] = []
            if not mesh["attr_normals_indices"]:
                mesh["attr_normals_indices"] = []
            if mesh["normals"]:
                mesh["normals"][:] = [world_rot.TransformDir(x).GetNormalized() for x in mesh["normals"]]
            else:
                mesh["normals"] = []
                carb.log_warn(f"mesh doesn't contain normals: ({child_prim.GetName()})")
            if mesh["attr_normals"]:
                mesh["attr_normals"][:] = [world_rot.TransformDir(x) for x in mesh["attr_normals"]]
            mesh["vertex_counts"] = usdMesh.GetFaceVertexCountsAttr().Get()
            mesh["vertex_indices"] = usdMesh.GetFaceVertexIndicesAttr().Get()
            # mesh["st"] = usdMesh.GetPrimvar("st").Get()
            mesh["name"] = child_prim.GetName()
            mat, rel = UsdShade.MaterialBindingAPI(usdMesh).ComputeBoundMaterial()
            mat_path = str(mat.GetPath())
            # if self.override_looks_directory[0].get_value_as_bool():
            #     _mat_path = "{}/{}".format(
            #         self.override_looks_directory[1].get_value_as_string(), mat_path.rsplit("/", 1)[-1]
            #     )
            #     if stage.GetPrimAtPath(_mat_path):
            #         mat_path = _mat_path
            #     else:
            #         carb.log_error(f"Overridden material not found ({mat_path})")
            if not rel:
                mat_path = "/None"
            # if rel:
            #     mesh["mat"] = str(mat.GetPath())
            # else:
            mesh["mat"] = mat_path
            subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(child_prim))
            mesh["subset"] = []
            for s in subsets:
                mat, rel = UsdShade.MaterialBindingAPI(s).ComputeBoundMaterial()
                mat_path = str(mat.GetPath())
                # if self.override_looks_directory[0].get_value_as_bool():
                #     _mat_path = "{}/{}".format(
                #         self.override_looks_directory[1].get_value_as_string(), mat_path.rsplit("/", 1)[-1]
                #     )
                #     if stage.GetPrimAtPath(_mat_path):
                #         mat_path = _mat_path
                if not rel:
                    mat_path = "/None"
                mesh["subset"].append((mat_path, s.GetIndicesAttr().Get()))
            # print(mat.GetPath(), rel)
            # print("INDICES", mesh["normals"])
            meshes.append(mesh)
            # print(count)
            # print(len(mesh["points"]), len(mesh["normals"]), len(mesh["vertex_counts"]), len(mesh["vertex_indices"]))
            count = count + 1
    carb.log_info(f"Merging: {count} meshes")
    all_points = []
    all_normals = []
    all_normals_attr = []
    all_normals_indices = []
    all_vertex_counts = []
    all_vertex_indices = []
    all_mats = {}
    index_offset = 0
    normals_offset = 0
    index = 0
    range_offset = 0
    for mesh in meshes:
        all_points.extend(mesh["points"])
        all_normals.extend(mesh["normals"])
        all_normals_attr.extend(mesh["attr_normals"])
        mesh["attr_normals_indices"][:] = [x + normals_offset for x in mesh["attr_normals_indices"]]
        all_normals_indices.extend(mesh["attr_normals_indices"])
        if mesh["normals"]:
            mesh["normals"][:] = [world_rot.TransformDir(x).GetNormalized() for x in mesh["normals"]]
        all_vertex_counts.extend(mesh["vertex_counts"])
        mesh["vertex_indices"][:] = [x + index_offset for x in mesh["vertex_indices"]]
        all_vertex_indices.extend(mesh["vertex_indices"])
        # all_st.extend(mesh["st"])
        index_offset = index_offset + len(meshes[index]["points"])
        normals_offset = normals_offset + len(mesh["attr_normals_indices"])
        # print("Offset", index_offset)
        index = index + 1
        # create the material entry
        if len(mesh["subset"]) == 0:
            if mesh["mat"] not in all_mats:
                all_mats[mesh["mat"]] = []
            all_mats[mesh["mat"]].extend([*range(range_offset, range_offset + len(mesh["vertex_counts"]), 1)])
        else:
            for subset in mesh["subset"]:
                if subset[0] not in all_mats:
                    all_mats[subset[0]] = []
                all_mats[subset[0]].extend([*(x + range_offset for x in subset[1])])
        range_offset = range_offset + len(mesh["vertex_counts"])
    merged_path = "/Merged/" + str(curr_prim.GetName())
    merged_path = omni.usd.get_stage_next_free_path(stage, merged_path, False)
    carb.log_info(f"Merging to path: {merged_path}")
    merged_mesh = UsdGeom.Mesh.Define(stage, merged_path)
    xform = UsdGeom.Xformable(merged_mesh)
    xform_op_t = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_r = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_t.Set(prim_transform.ExtractTranslation())
    q = prim_transform.ExtractRotation().GetQuaternion()
    xform_op_r.Set(Gf.Quatd(q.GetReal(), q.GetImaginary()))
    # xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
    # if not self.parent_xform.get_value_as_bool():
    # xform_op.Set(prim_transform)
    # merged_mesh.CreateSubdivisionSchemeAttr("none")
    # merged_mesh.CreateTriangleSubdivisionRuleAttr("smooth")
    merged_mesh.CreatePointsAttr(all_points)
    if all_normals:
        merged_mesh.CreateNormalsAttr(all_normals)
        merged_mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    merged_mesh.CreateSubdivisionSchemeAttr("none")
    merged_mesh.CreateFaceVertexCountsAttr(all_vertex_counts)
    merged_mesh.CreateFaceVertexIndicesAttr(all_vertex_indices)
    if all_normals_attr:
        normals_attr = merged_mesh.GetPrim().CreateAttribute("primvars:normals", Sdf.ValueTypeNames.Float3Array, False)
        normals_attr.Set(all_normals_attr)
        normals_attr.SetMetadata("interpolation", "vertex")
        merged_mesh.GetPrim().CreateAttribute("primvars:normals:indices", Sdf.ValueTypeNames.IntArray, False).Set(
            all_normals_indices
        )
    extent = merged_mesh.ComputeExtent(all_points)
    merged_mesh.CreateExtentAttr().Set(extent)
    # texCoord = merged_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    # texCoord.Set(all_st)
    # print(all_mats)
    for name, counts in sorted(all_mats.items(), key=lambda a: a[0].rsplit("/", 1)[-1]):
        subset_name = merged_path + "/{}".format(name.rsplit("/", 1)[-1])
        geomSubset = UsdGeom.Subset.Define(stage, omni.usd.get_stage_next_free_path(stage, subset_name, False))
        geomSubset.CreateElementTypeAttr("face")
        geomSubset.CreateFamilyNameAttr("materialBind")
        # print(mesh["vertex_indices"])
        geomSubset.CreateIndicesAttr(counts)
        if name != "/None":
            material = UsdShade.Material.Get(stage, name)
            binding_api = UsdShade.MaterialBindingAPI(geomSubset)
            binding_api.Bind(material)


def _add_terrain_border(bbox, border_width):
    """Add a surrounding border over all the sub-terrains into the terrain meshes."""
    bbox_dim = np.sum(np.abs(bbox), axis=0)
    border_size = (bbox_dim[0] + 2 * border_width, bbox_dim[1] + 2 * border_width)
    border_center = (np.mean(bbox[:, 0]), np.mean(bbox[:, 1]), 0.5)
    return trimesh.util.concatenate(
        make_border(border_size, [bbox_dim[0], bbox_dim[1]], height=3.0, position=border_center)
    )


def main():
    # convert any .obj files to .usd
    converter_cfg = sim_utils.MeshConverterCfg(
        mass_props=sim_utils.MassPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        asset_path=args_cli.env_list[0],
        usd_dir=os.path.dirname(args_cli.env_list[0]),
        usd_file_name=os.path.basename(args_cli.env_list[0]).replace(".obj", ".usd"),
        make_instanceable=False,
        rotation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    for env in args_cli.env_list:
        sim_utils.MeshConverter(
            converter_cfg.replace(
                asset_path=env,
                usd_file_name=os.path.basename(env).replace(".obj", ".usd"),
                usd_dir=os.path.dirname(env),
            )
        )

    # init simulation context
    sim_utils.SimulationContext()

    # init terrain importer
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=len(args_cli.env_list),
        prim_path="/World",
        env_spacing=1.0,
        terrain_type="usd",
        usd_path=args_cli.env_list[0].replace(".obj", ".usd"),
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)
    # import the terrain
    for i, env in enumerate(args_cli.env_list[1:]):
        terrain_importer.import_usd(f"terrain_{i}", env.replace(".obj", ".usd"))
    # configure the origins in a grid
    prim = prim_utils.get_prim_at_path(os.path.join(terrain_importer_cfg.prim_path, f"terrain_{i}", "geometry", "mesh"))
    extent = np.array(prim.GetAttribute("extent").Get())
    terrain_importer_cfg.env_spacing = np.sum(np.abs(extent[:, 0]))
    terrain_importer.configure_env_origins()
    # move the terrains to their origins
    for i, terrain_prim in enumerate(prim_utils.get_prim_children(prim_utils.get_prim_at_path("/World"))):
        terrain_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(terrain_importer.env_origins[i].tolist()))

    # add border to terrains
    mean_terrain_size = np.mean(np.abs(extent), axis=0)
    terrain_max = terrain_importer.env_origins.max(dim=0)[0].tolist()
    terrain_min = terrain_importer.env_origins.min(dim=0)[0].tolist()
    bbox = np.vstack((terrain_max + mean_terrain_size, terrain_min - mean_terrain_size))
    terrain_importer.import_mesh("border", _add_terrain_border(bbox, border_width=1.0))

    # merge imported meshes
    _merge_mesh(prim_utils.get_prim_at_path("/World"))

    # add collision properties
    sim_utils.define_collision_properties("/Merged", sim_utils.CollisionPropertiesCfg(collision_enabled=True))
    # export merged prim to file
    sim_utils.export_prim_to_file(
        source_prim_path="/Merged",
        target_prim_path="/NavigationTerrain",
        path=os.path.join(os.path.dirname(os.path.dirname(args_cli.env_list[0])), "merged_mesh.usd"),
    )

    print("Done merging")


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
