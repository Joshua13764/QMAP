from pathlib import Path
from typing import Any, List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.steps.PAN_to_LOD import PANToLOD
from boulder_statistics.steps.PAN_to_LOD_supersample import PANToLODSuperSample
from boulder_statistics.steps.pds4_boulderNet_inference import \
    PDS4BoulderNetInference
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

detections_from_bennu_pan: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\detections_from_bennu_pan")

get_pan = SimpleRequest(
    task_name=f"Downloader for the bennu PAN file",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
    fs_path=detections_from_bennu_pan.as_posix(),
    sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
    markers=(FSMarkerString(value="PAN_texture"),)
)

super_sample_steps = []
super_sample_factors: list[int] = [1, 2, 4, 8]
for factor in super_sample_factors:
    super_sample_steps.append(
        PANToLODSuperSample(
            task_name=f"Convert bennu PAN to LODs - Numpy version (super sample x{factor})",
            root_path=detections_from_bennu_pan,
            run_after_task_names=(get_pan.task_name,),
            supersample_factor=factor,
            lod_res=512,
            skip_if_exists=True,
            import_markers=(FSMarkerString(value="PAN_texture"),),
            export_markers=(FSMarkerString(
                value=f"InferableImage"),),
            extract_folder_prefix=f"PAN_lod_np (super sample x{factor})",
            lod_depth=5,
            export_adapter=FSNumpyAdapter(
                export_debug_plots=True,
                title=f"PAN to LOD NP with super sample x{factor}",
                colour_bar_title="Pixel Value",
            )
        ))

boulder_detections = PDS4BoulderNetInference(
    task_name=f"Infer boulders on bennu PAN LODs with BoulderNet -",
    cuda=True,
    skip_converted=True,
    run_after_task_names=tuple(
        [step.task_name for step in super_sample_steps]),
    run_path=detections_from_bennu_pan.as_posix(),
    detection_input_markers=(FSMarkerString("InferableImage"),),
    detection_output_markers=(FSMarkerString("BoulderNet_Detections"),)
)

steps: List[Any] = [get_pan, *super_sample_steps, boulder_detections]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
