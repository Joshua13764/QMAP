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
from boulder_statistics.file_storage_adapters.OCAMS_image_as_mesh_adapter import \
    FSOCAMSImageAsMeshAdapter
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
from boulder_statistics.file_storage_adapters.tiff_adapter import FSTiffAdapter
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

las_export_adapter: FSBennuObjToLODCubemapGeneratorAdapter = FSBennuObjToLODCubemapGeneratorAdapter(
    # FSNumpyAdapter(export_debug_plots=True, title="Export Projection",
    # colour_bar_title="OCAMS colour value", transform=lambda x: x), n_jobs=1)
    tiles_adapter=FSTiffAdapter(export_as_jpeg=True)
)


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

# Bulk read form manifest (df) after filtering

export_las = BetterOBJToLAS(
    task_name="divide las into lods cool",
    lod_depth=1,
    lod_res=512,
    run_after_task_names=(get_bennu_obj.task_name,),
    colour_column_name=lambda face: f'color',
    input_adapter=FSOCAMSImageAsMeshAdapter(
        positions_path=r"C:\Users\Joshu\OneDrive - Nexus365\AO33\Boulder_database\Investigations\project_from_ocams_image\20190926T173446S252_map_specradL2b_IAU_BENNU_positions.tiff",
        colors_path=r"C:\Users\Joshu\OneDrive - Nexus365\AO33\Boulder_database\Investigations\project_from_ocams_image\20190926T173446S252_map_specradL2b_ocams.tiff"
    ),
    output_adapter=las_export_adapter,
    input_markers=(FSMarkerString(value="OCAMS_Model"),),
    output_markers=(FSMarkerString(value="LAS_lod"),),
    debug_mode=True,
    pipeline_data_path=Path(
        r"C:\Users\Joshu\OneDrive - Nexus365\AO33\Boulder_database\Investigations\project_from_ocams_image\export"),
    n_jobs=1,
)

steps: List[Any] = [
    export_las,
    get_bennu_obj
]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
