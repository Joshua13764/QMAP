from pathlib import Path
from typing import Any, List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.file_storage_adapters.fs_copy_cubemap_generator_adapter import \
    FSCopyCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.fs_cubemap_generator_adapter import \
    FSCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.iio_adapter import FSIIOAdapter
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.file_storage_adapters.pan_to_lod_cubemap_generator_adapter import \
    FSPANToLODCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.shutil_copy_adapter import \
    FSShutilCopyAdapter
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.steps.Better_PAN_to_LOD import BetterPANToLOD
from boulder_statistics.steps.better_PDS4_boulder_net_inference import \
    BetterPDS4BoulderNetInference
from boulder_statistics.steps.PAN_to_LOD import PANToLOD
from boulder_statistics.steps.PAN_to_LOD_supersample import PANToLODSuperSample
from boulder_statistics.steps.pds4_boulderNet_inference import \
    PDS4BoulderNetInference
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

detections_from_bennu_pan: Path = Path(
    r"C:\Users\Joshu\OneDrive - Nexus365\AO33\Testing\Extract detections")

get_pan = SimpleRequest(
    task_name=f"Downloader for the bennu PAN",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
    fs_path=detections_from_bennu_pan.as_posix(),
    sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
    markers=(FSMarkerString(value="PAN_texture"),)
)

lod_export_adapter: FSPANToLODCubemapGeneratorAdapter = FSPANToLODCubemapGeneratorAdapter(
    tiles_adapter=FSNumpyAdapter(export_debug_plots=True, title="Export lod", colour_bar_title="colour value"), n_jobs=18)

divide_pan: BetterPANToLOD = BetterPANToLOD(
    task_name="divide pan into lods",
    run_after_task_names=(get_pan.task_name,),
    input_adapter=FSIIOAdapter(),
    output_adapter=lod_export_adapter,
    input_markers=(FSMarkerString(value="PAN_texture"),),
    output_markers=(FSMarkerString(value="PAN_lod"),),
    pipeline_data_path=detections_from_bennu_pan,
    n_jobs=1,
)

detection = BetterPDS4BoulderNetInference(
    task_name=f"Infer boulders on bennu PAN LODs with BoulderNet",
    run_after_task_names=(divide_pan.task_name,),
    cuda=True,
    input_adapter=FSCopyCubemapGeneratorAdapter(),
    input_markers=(FSMarkerString(value="PAN_lod"),),
    output_markers=(FSMarkerString(value="INF_lod"),),
    pipeline_data_path=detections_from_bennu_pan,
)

# boulder_detections = PDS4BoulderNetInference(
#     task_name=f"Infer boulders on bennu PAN LODs with BoulderNet -",
#     cuda=True,
#     skip_converted=True,
#     run_after_task_names=tuple(
#         [step.task_name for step in super_sample_steps]),
#     run_path=detections_from_bennu_pan.as_posix(),
#     detection_input_markers=(FSMarkerString("InferableImage"),),
#     detection_output_markers=(FSMarkerString("BoulderNet_Detections"),),
#     append_input_extension_no_dot="npy",
# )

steps: List[Any] = [get_pan, divide_pan, detection]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
