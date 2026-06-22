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
from boulder_statistics.step_base import StepBase
from boulder_statistics.steps.Better_OBJ_to_LAS import BetterOBJToLAS
from boulder_statistics.steps.Better_PAN_to_LOD import BetterPANToLOD
from boulder_statistics.steps.better_PDS4_boulder_net_inference import \
    BetterPDS4BoulderNetInference
from boulder_statistics.steps.export_full_data_pack import ExportFullDataPack
from boulder_statistics.steps.setup_boulder_net_inferences_for_grading import \
    SetupBoulderNetInferencesForGrading
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator


class StandardInferRun():
    @staticmethod
    def get_infer_steps(pipeline_cache_path: Path,
                        data_pack_export_folder: Path) -> List[StepBase]:

        detections_from_bennu_pan: Path = pipeline_cache_path / Path(
            r"G:\AO33_pipeline_folders\LAS export + 30 detect threshold + D4_transforms")

        get_pan = SimpleRequest(
            task_name=f"Downloader for the bennu PAN",
            url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
            fs_path=detections_from_bennu_pan.as_posix(),
            sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
            markers=(FSMarkerString(value="PAN_texture"),)
        )

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

        las_export_adapter: FSBennuObjToLODCubemapGeneratorAdapter = FSBennuObjToLODCubemapGeneratorAdapter(
            tiles_adapter=FSNumpyAdapter(export_debug_plots=True, title="Export LAS", colour_bar_title="1 / LAS factor", transform=lambda x: 1 / x))

        export_las = BetterOBJToLAS(
            task_name="divide las into lods",
            lod_depth=4,
            run_after_task_names=(get_bennu_obj.task_name,),
            input_adapter=FSPolarsObjAdapterFastPLOBJData(),
            output_adapter=las_export_adapter,
            input_markers=(FSMarkerString(value="OCAMS_Model"),),
            output_markers=(FSMarkerString(value="LAS_lod"),),
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
            n_jobs=1,
        )

        grades = SetupBoulderNetInferencesForGrading(
            # debug_mode=True,
            task_name=f"Setup inferences for grading",
            run_after_task_names=(
                detection.task_name,
                divide_pan.task_name,
                export_las.task_name),
            pipeline_data_path=detections_from_bennu_pan,
            output_markers=(FSMarkerString(value="INFCOLL_lod"),),
            output_adapter=FSPickleAdapter(),
            lod_images_input=FSInput(
                fs_marker=FSMarkerString(value="PAN_lod"),
                fs_adapter=FSGenericCubemapGeneratorAdapter(
                    tiles_adapter=FSNumpyAdapter())
            ),
            LAS_factor_input=FSInput(
                fs_marker=FSMarkerString(value="LAS_lod"),
                fs_adapter=FSGenericCubemapGeneratorAdapter(
                    tiles_adapter=FSNumpyAdapter())
            ),
            lod_detections_input=FSInput(
                fs_marker=FSMarkerString(value="INF_lod"),
                fs_adapter=FSGenericCubemapGeneratorAdapter(
                    tiles_adapter=FSInferenceDetectionAdapter())
            ),
            input_markers=None
        )

        export_detections = ExportFullDataPack(
            task_name=f"Export full data pack to DF",
            run_after_task_names=(grades.task_name, ),
            pipeline_data_path=detections_from_bennu_pan,
            input_adapter=FSTypeSafePickleAdapter(
                expected_type=ImageDetectionGrades),
            output_adapter=FSPolarsLazyActionBatched(
                temp_folder_path=data_pack_export_folder.as_posix(),
                n_jobs=4,
                export_lazy_frame_adapter=FSPolarsLazyCSVAdapter(),
                standard_extension="csv"),
            input_markers=(FSMarkerString(value="INFCOLL_lod"),),
            output_markers=(FSMarkerString(value="GRAD_lod_export"),),
        )

        steps: List[Any] = [
            get_pan,
            divide_pan,
            detection,
            grades,
            export_detections,
            export_las,
            get_bennu_obj
        ]

        return steps

# How to use
# if __name__ == "__main__":
#     cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
#         cache_folder=Path(".cache"), result_type=FSEnvironment)

#     futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
#         steps, cache)
