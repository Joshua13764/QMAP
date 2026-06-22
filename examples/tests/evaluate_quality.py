from pathlib import Path
from typing import List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.step_base import StepBase
from boulder_statistics.steps.Best_model_downloader import BestModelDownloader
from boulder_statistics.steps.detection_merge import DetectionMerge
from boulder_statistics.steps.OBJ_to_LAS import OBJToLAS
from boulder_statistics.steps.PAN_to_LOD import PANToLOD
from boulder_statistics.steps.pds4_boulderNet_inference import \
    PDS4BoulderNetInference
from boulder_statistics.steps.PDS_downloader import PDSDownloader
from boulder_statistics.steps.PDS_to_PNG import PDS_to_PNG
from boulder_statistics.steps.plot_standard_detection_results import \
    PlotStandardDetectionResults
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

pds_download_path: Path = Path(
    r"C:\Users\Joshu\OneDrive - Nexus365\AO33\Boulder_database\Investigations\BoulderNet\examples\pipeline_work_folder")
pipeline_working_path_fast: Path = Path(
    r"C:\Users\Joshu\OneDrive - Nexus365\AO33\Boulder_database\Investigations\BoulderNet\examples\section_export")

step2 = SimpleRequest(
    task_name=f"Downloader for the bennu PAN file",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
    fs_path=pds_download_path.as_posix(),
    sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
    markers=(FSMarkerString(value="PAN_texture"),)
)

pan_to_lod_np = PANToLOD(
    task_name=f"Convert bennu PAN to LODs - Numpy version",
    root_path=pipeline_working_path_fast,
    run_after_task_names=(step2.task_name,),
    lod_res=8192,
    skip_if_exists=True,
    import_markers=(FSMarkerString(value="PAN_texture"),),
    export_markers=(FSMarkerString(value="PAN_lod_np"),),
    extract_folder_prefix="PAN_lod_np",
    lod_depth=1,
    export_adapter=FSNumpyAdapter()
)

steps = [
    step2,
    pan_to_lod_np]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_tasks_with_dependencies(
        [pan_to_lod_np], steps, cache)
