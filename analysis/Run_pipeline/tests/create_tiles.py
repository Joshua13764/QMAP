from pathlib import Path
from typing import Any, List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.file_storage_adapters.bennu_obj_to_las_cubemap_generator_adapter import \
    FSBennuObjToLODCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.polars_obj_adapter_fast_PL_obj_data_specify_path import \
    FSPolarsObjAdapterFastPLOBJDataSpecifyPath
from boulder_statistics.file_storage_adapters.tiff_adapter import FSTiffAdapter
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.steps.Better_OBJ_to_LAS import BetterOBJToLAS
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

pipeline_data_path = Path(r"G:\AO33\Data products\tiles\runs_NOT_INCLUDE")

UHD_mesh = Path(
    r"G:\AO33\Data products\tiles\meshes\bennu_OLA_v21_PTM_very-high.obj")
HD_mesh = Path(
    r"G:\AO33\Data products\tiles\meshes\g_00880mm_alt_ptm_0000n00000_v020.obj")
HD_smoothed_mesh = Path(
    r"G:\AO33\Data products\tiles\meshes\g_00880mm_alt_ptm_0000n00000_v020_smoothed.obj")

detections_from_bennu_pan: Path = Path(
    r"G:\AO33_pipeline_folders\LAS export + 30 detect threshold")

get_bennu_obj = SimpleRequest(
    task_name=f"Downloader for the bennu OBJ (LQ) mesh",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/g_00880mm_alt_ptm_0000n00000_v020.obj",
    fs_path=detections_from_bennu_pan.as_posix(),
    sub_path=Path(
        "OCAMS",
        "Global Bennu 3D model - OLA v20 PTM.obj").as_posix(),
    markers=(
        FSMarkerString(
            value="OCAMS_Model"),)
)

tasks = []

# ====================================== TASKS ===========================

tasks.append(BetterOBJToLAS(
    task_name="LAS Very High Poly",
    lod_depth=4,
    run_after_task_names=(get_bennu_obj.task_name,),
    colour_column_name=lambda face: f'{face}_ratio',
    input_adapter=FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
        mesh_path=UHD_mesh.as_posix()
    ),
    output_adapter=FSBennuObjToLODCubemapGeneratorAdapter(
        tiles_adapter=FSTiffAdapter(
            export_as_jpeg=True,
            export_function=lambda x: 1 / x)
    ),
    input_markers=(FSMarkerString(value="OCAMS_Model"),),
    output_markers=(FSMarkerString(value="LAS_lod"),),
    pipeline_data_path=pipeline_data_path,
    n_jobs=1,
))

tasks.append(BetterOBJToLAS(
    task_name="LAS High Poly",
    lod_depth=4,
    run_after_task_names=(get_bennu_obj.task_name,),
    colour_column_name=lambda face: f'{face}_ratio',
    input_adapter=FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
        mesh_path=HD_mesh.as_posix()
    ),
    output_adapter=FSBennuObjToLODCubemapGeneratorAdapter(
        tiles_adapter=FSTiffAdapter(
            export_as_jpeg=True,
            export_function=lambda x: 1 / x)
    ),
    input_markers=(FSMarkerString(value="OCAMS_Model"),),
    output_markers=(FSMarkerString(value="LAS_lod"),),
    pipeline_data_path=pipeline_data_path,
    n_jobs=1,
))

tasks.append(BetterOBJToLAS(
    task_name="LAS High Poly Smoothed",
    lod_depth=4,
    run_after_task_names=(get_bennu_obj.task_name,),
    colour_column_name=lambda face: f'{face}_ratio',
    input_adapter=FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
        mesh_path=HD_smoothed_mesh.as_posix()
    ),
    output_adapter=FSBennuObjToLODCubemapGeneratorAdapter(
        tiles_adapter=FSTiffAdapter(
            export_as_jpeg=True,
            export_function=lambda x: 1 / x)
    ),
    input_markers=(FSMarkerString(value="OCAMS_Model"),),
    output_markers=(FSMarkerString(value="LAS_lod"),),
    pipeline_data_path=pipeline_data_path,
    n_jobs=1,
))

for axis in ["x", "y", "z"]:
    tasks.append(BetterOBJToLAS(
        task_name=f"{axis} position Very High Poly",
        lod_depth=4,
        run_after_task_names=(get_bennu_obj.task_name,),
        colour_column_name=lambda face: f'{axis}_tri_mean',
        input_adapter=FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
            mesh_path=UHD_mesh.as_posix()
        ),
        output_adapter=FSBennuObjToLODCubemapGeneratorAdapter(
            tiles_adapter=FSTiffAdapter(
                export_as_jpeg=True)
        ),
        input_markers=(FSMarkerString(value="OCAMS_Model"),),
        output_markers=(FSMarkerString(value="LAS_lod"),),
        pipeline_data_path=pipeline_data_path,
        n_jobs=1,
    ))

    tasks.append(BetterOBJToLAS(
        task_name=f"{axis} position High Poly",
        lod_depth=4,
        run_after_task_names=(get_bennu_obj.task_name,),
        colour_column_name=lambda face: f'{axis}_tri_mean',
        input_adapter=FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
            mesh_path=HD_mesh.as_posix()
        ),
        output_adapter=FSBennuObjToLODCubemapGeneratorAdapter(
            tiles_adapter=FSTiffAdapter(
                export_as_jpeg=True)
        ),
        input_markers=(FSMarkerString(value="OCAMS_Model"),),
        output_markers=(FSMarkerString(value="LAS_lod"),),
        pipeline_data_path=pipeline_data_path,
        n_jobs=1,
    ))

    tasks.append(BetterOBJToLAS(
        task_name=f"{axis} position High Poly Smoothed",
        lod_depth=4,
        run_after_task_names=(get_bennu_obj.task_name,),
        colour_column_name=lambda face: f'{axis}_tri_mean',
        input_adapter=FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
            mesh_path=HD_smoothed_mesh.as_posix()
        ),
        output_adapter=FSBennuObjToLODCubemapGeneratorAdapter(
            tiles_adapter=FSTiffAdapter(
                export_as_jpeg=True)
        ),
        input_markers=(FSMarkerString(value="OCAMS_Model"),),
        output_markers=(FSMarkerString(value="LAS_lod"),),
        pipeline_data_path=pipeline_data_path,
        n_jobs=1,
    ))

# ====================================== TASKS ===========================

steps: List[Any] = [
    *tasks,
    get_bennu_obj
]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
