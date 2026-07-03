from pathlib import Path
from typing import Any, List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_input import FSInput
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.file_storage_adapters.bennu_obj_to_las_cubemap_generator_adapter import \
    FSBennuObjToLODCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.fs_copy_cubemap_generator_adapter import \
    FSCopyCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.fs_generic_cubemap_generator_adapter import \
    FSGenericCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.iio_adapter import FSIIOAdapter
from boulder_statistics.file_storage_adapters.inference_detection_adapter import \
    FSInferenceDetectionAdapter
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.file_storage_adapters.pan_to_lod_cubemap_generator_adapter import \
    FSPANToLODCubemapGeneratorAdapter
from boulder_statistics.file_storage_adapters.pickle_adapter import \
    FSPickleAdapter
from boulder_statistics.file_storage_adapters.polars_lazy_action_csv_batched_adapter import \
    FSPolarsLazyActionBatched
from boulder_statistics.file_storage_adapters.polars_lazy_csv_adapter import \
    FSPolarsLazyCSVAdapter
from boulder_statistics.file_storage_adapters.polars_obj_adapter_fast_PL_obj_data import \
    FSPolarsObjAdapterFastPLOBJData
from boulder_statistics.file_storage_adapters.type_safe_pickle_adapter import \
    FSTypeSafePickleAdapter
from boulder_statistics.lods.utils.image_detection_grades import \
    ImageDetectionGrades
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.steps.Better_OBJ_to_LAS import BetterOBJToLAS
from boulder_statistics.steps.Better_PAN_to_LOD import BetterPANToLOD
from boulder_statistics.steps.better_PDS4_boulder_net_inference import \
    BetterPDS4BoulderNetInference
from boulder_statistics.steps.export_full_data_pack import ExportFullDataPack
from boulder_statistics.steps.setup_boulder_net_inferences_for_grading import \
    SetupBoulderNetInferencesForGrading
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

detections_from_bennu_pan: Path = Path(
    r"G:\AO33_pipeline_folders\LAS export + 30 detect threshold")

get_pan = SimpleRequest(
    task_name=f"Downloader for the bennu PAN",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
    fs_path=detections_from_bennu_pan.as_posix(),
    sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
    markers=(FSMarkerString(value="PAN_texture"),)
)

lod_export_adapter: FSPANToLODCubemapGeneratorAdapter = FSPANToLODCubemapGeneratorAdapter(
    tiles_adapter=FSNumpyAdapter(export_debug_plots=True, title="Export lod", colour_bar_title="colour value"), n_jobs=4)

divide_pan: BetterPANToLOD = BetterPANToLOD(
    task_name="divide pan into lods",
    lod_depth=4,
    run_after_task_names=(get_pan.task_name,),
    input_adapter=FSIIOAdapter(),
    output_adapter=lod_export_adapter,
    input_markers=(FSMarkerString(value="PAN_texture"),),
    output_markers=(FSMarkerString(value="PAN_lod"),),
    pipeline_data_path=detections_from_bennu_pan,
    n_jobs=1,
)


steps: List[Any] = [
    get_pan,
    divide_pan,
    get_bennu_obj
]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
