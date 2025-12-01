# Run "prefect server start" to start server before running this script
import inspect
import types
from pathlib import Path
from typing import Any, Coroutine, List, Sequence

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor.step_templates.simple_local_file import \
    SimpleLocalFile
from bennu_feature_extractor.step_templates.simple_request import SimpleRequest
from bennu_feature_extractor.steps_orchestrator import StepsOrchestrator
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_BoulderNet.detection_merge import DetectionMerge
from bennu_feature_extractor_BoulderNet.pds4_boulderNet_inference import \
    PDS4BoulderNetInference
from bennu_feature_extractor_BoulderNet.plot_standard_detection_results import \
    PlotStandardDetectionResults
from bennu_feature_extractor_PDS.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from bennu_feature_extractor_PDS.OBJ_to_LAS import OBJToLAS
from bennu_feature_extractor_PDS.PAN_to_LOD import PANToLOD
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from bennu_feature_extractor_PDS.PDS_to_PNG import PDS_to_PNG
from bennu_feature_extractor_PDS.SPICE_kernels_downloader import \
    SPICEKernelGrabber
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture

try:
    RES_STORE: LocalFileSystem | Coroutine[Any, Any,
                                           LocalFileSystem] = LocalFileSystem.load("run-dir-storage")
except ValueError:
    RES_STORE = LocalFileSystem(basepath=".\\.run_dir_storage")
    RES_STORE.save("run-dir-storage", overwrite=True)


model_download_path: Path = Path(r"F:\AO33\AO33_models")
pds_download_path: Path = Path(r"F:\AO33\AO33_pds_DATA")
pipeline_working_path: Path = Path(r"F:\AO33\AO33_pipeline_DATA")
pipeline_working_path_fast: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\Pipeline_running_path_fast")
boulderNet_preprocessor_test_folder: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\BoulderNetPreProcessorTests"
)
spice_download_path: Path = Path(r"F:\AO33\AO33_SPICE_DATA")

step1 = BestModelDownloader(
    task_name="Download the best boulderNet model",
    DownloadPath=model_download_path.as_posix(),
    Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
)

step2 = SimpleRequest(
    task_name=f"Downloader for the bennu PAN file",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
    fs_path=pds_download_path.as_posix(),
    sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
    markers=frozenset([FSMarkerString(value="PAN_texture")])
)

step3 = SimpleRequest(
    task_name=f"Downloader for the bennu OBJ (LQ) mesh",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/g_00880mm_alt_ptm_0000n00000_v020.obj",
    fs_path=pds_download_path.as_posix(),
    sub_path=Path(
        "OCAMS",
        "Global Bennu 3D model - OLA v20 PTM.obj").as_posix(),
    markers=frozenset(
        [FSMarkerString(value="OCAMS Model"), FSMarkerString("ProjectModel")])
)

steps4: List[PDSDownloader] = [PDSDownloader(
    task_name=f"Downloader for PDS file {url}",
    DownloadPath=pds_download_path.as_posix(),
    Url=url
)
    for url in [
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_reduced_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_orbit_b.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_recon.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_metadata.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_calibration.zip"]
]

step5 = PDS_to_PNG(
    task_name=f"Convert cluster ocams_data_calibrated_detailed_survey",
    run_after_task_names=frozenset([steps4[0].task_name]),
    cluster_key="ocams_data_calibrated_detailed_survey",
    run_path=pipeline_working_path.as_posix()
)

step6 = PANToLOD(
    task_name=f"Convert bennu PAN to LODs",
    root_path=pipeline_working_path_fast,
    run_after_task_names=frozenset([step2.task_name]),
    extract_folder_prefix="PAN_lod_default",
    lod_res=512,
    lod_depth=6,
    skip_if_exists=True
)

pan_to_lod_np = PANToLOD(
    task_name=f"Convert bennu PAN to LODs - Numpy version",
    root_path=pipeline_working_path_fast,
    run_after_task_names=frozenset([step2.task_name]),
    lod_res=512,
    skip_if_exists=True,
    export_markers=frozenset([FSMarkerString(value="PAN_lod_np")]),
    extract_folder_prefix="PAN_lod_np",
    lod_depth=6,
    export_adapter=FSNumpyAdapter()
)

step7 = OBJToLAS(
    task_name=f"Convert bennu Mesh to stretch maps",
    run_after_task_names=frozenset([step3.task_name]),
    lod_res=512,
    export_folder=pipeline_working_path_fast.as_posix(),
    depth=6,
    skip_if_exists=True,
    debug_mode=False
)

step8 = PDS4BoulderNetInference(
    task_name=f"Infer boulders",
    cuda=True,
    skip_converted=True,
    run_after_task_names=frozenset([step6.task_name, step1.task_name]),
    run_path=pipeline_working_path_fast.as_posix(),
    detection_output_markers=frozenset(
        [FSMarkerString("BoulderNet_Detections")])
)

step10 = DetectionMerge(
    task_name=f"Merge detections",
    run_after_task_names=frozenset([step8.task_name]),
    marker_to_merge=FSMarkerString("BoulderNet_Detections"),
    output_marker=FSMarkerString("Merged_BoulderNet_Detections"),
    run_path=pipeline_working_path_fast.as_posix(),
    result_output_path=Path("exports/merge_detections.pkl").as_posix()
)

step11 = PlotStandardDetectionResults(
    task_name=f"Plot standard detection results",
    run_after_task_names=frozenset([step10.task_name]),
    marker_to_plot=FSMarkerString("Merged_BoulderNet_Detections"),
    output_marker=FSMarkerString("Detection_Plots"),
    export_folder=pipeline_working_path_fast.as_posix(),
    result_output_folder=Path("exports/detection_plots").as_posix(),
    version_index=3
)

# step9 = SPICEKernelGrabber(
#     task_name=f"Collect SPICE kernels",
#     DownloadPath=spice_download_path.as_posix(),
#     MkUrls=(
#         "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/mk/orx_2019_v08.tm",
#     ),
#     ExtraUrls=("https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/dsk/bennu_g_00880mm_alt_obj_0000n00000_v021a.bds",),
# )

default_lod_extract = SimpleLocalFile(
    task_name="BoulderNet detect tests",
    run_after_task_names=frozenset(),
    local_file_path=Path(
        r"C:\Users\Joshu\Documents\AO33_DATA\resources\tests\posx_512_1024_512x512_of_2048.png"),
    dst_root_path=boulderNet_preprocessor_test_folder,
    dst_sub_path=Path("tests/default.png"),
    markers=frozenset([FSMarkerString("default_lod_export_example")])
)

infer_default_lod_extract = PDS4BoulderNetInference(
    task_name=f"Infer default lod_extract",
    cuda=True,
    skip_converted=True,
    run_after_task_names=frozenset([default_lod_extract.task_name]),
    run_path=boulderNet_preprocessor_test_folder.as_posix(),
    detection_input_markers=frozenset(
        [FSMarkerString("default_lod_export_example")]),
    detection_output_markers=frozenset(
        [FSMarkerString("BoulderNet_Detections")]),
)

futures: dict[str, PrefectFuture[FSEnvironment]
              ] = StepsOrchestrator.run_tasks_with_dependencies([infer_default_lod_extract], StepsOrchestrator.auto_find_steps(), RES_STORE)
