from pathlib import Path

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.steps.OBJ_to_LAS import OBJToLAS
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

detections_from_bennu_model: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\detections_from_bennu_model")

get_bennu_obj = SimpleRequest(
    task_name=f"Downloader for the bennu OBJ (LQ) mesh",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/g_00880mm_alt_ptm_0000n00000_v020.obj",
    fs_path=detections_from_bennu_model.as_posix(),
    sub_path=Path(
        "OCAMS",
        "Global Bennu 3D model - OLA v20 PTM.obj").as_posix(),
    markers=(
        FSMarkerString(
            value="OCAMS Model"),
        FSMarkerString("ProjectModel"))
)

get_local_area_scaling_lods = OBJToLAS(
    task_name=f"Convert bennu Mesh to stretch maps",
    run_after_task_names=(get_bennu_obj.task_name,),
    lod_res=512,
    export_folder=detections_from_bennu_model.as_posix(),
    depth=6,
    skip_if_exists=True,
    debug_mode=False
)

steps = [get_bennu_obj, get_local_area_scaling_lods]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_tasks_with_dependencies(
        [get_local_area_scaling_lods], steps, cache)
