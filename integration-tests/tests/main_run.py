from pathlib import Path
from typing import Any, Coroutine, List, Sequence

from bennu_feature_extractor.BFE_driver import BFEDriver
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor.step_templates.simple_request import SimpleRequest
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_BoulderNet.pds4_boulderNet_inference import \
    PDS4BoulderNetInference
from bennu_feature_extractor_PDS.OBJ_to_LAS import OBJToLAS
from bennu_feature_extractor_PDS.PAN_to_LOD import PANToLOD
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from bennu_feature_extractor_PDS.PDS_to_PNG import PDS_to_PNG
from bennu_feature_extractor_PDS.SPICE_kernels_downloader import \
    SPICEKernelGrabber
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture

# # To be run once
# run_dir_store = LocalFileSystem(basepath=".\\.run_dir_storage")
# run_dir_store.save("run-dir-storage", overwrite=True)

RES_STORE: LocalFileSystem | Coroutine[Any, Any,
                                       LocalFileSystem] = LocalFileSystem.load("run-dir-storage")

model_download_path: Path = Path(r"F:\AO33\AO33_models")
pds_download_path: Path = Path(r"F:\AO33\AO33_pds_DATA")
pipeline_working_path: Path = Path(r"F:\AO33\AO33_pipeline_DATA")
pipeline_working_path_fast: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\Pipeline_running_path_fast")
spice_download_path: Path = Path(r"F:\AO33\AO33_SPICE_DATA")

# Published 2019 metakernel (covers Detailed Survey, Orbit B, Recon in 2019)
MK_URLS: List[str] = [
    "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/mk/orx_2019_v08.tm",
]

# Optional: add a Bennu DSK as a plain file (no extraction)
EXTRA_URLS: List[str] = [
    # "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/dsk/bennu_g_00880mm_alt_obj_0000n00000_v021a.bds"
]

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
    root_path=pipeline_working_path.as_posix(),
    run_after_task_names=frozenset([step2.task_name]),
    lod_res=1024,
    skip_if_exists=True
)

step7 = OBJToLAS(
    task_name=f"Convert bennu Mesh to stretch maps",
    run_after_task_names=frozenset([step3.task_name]),
    lod_res=1024,
    depth=4,
    skip_if_exists=True,
    debug_mode=True
)

step8 = PDS4BoulderNetInference(
    task_name=f"Infer boulders",
    run_after_task_names=frozenset([step6.task_name, step1.task_name]),
    run_path=pipeline_working_path_fast
)

# step9 = SPICEKernelGrabber(
#     task_name=f"Collect SPICE kernels",
#     DownloadPath=spice_download_path.as_posix(),
#     MkUrls=MK_URLS,
#     ExtraUrls=EXTRA_URLS,
# )

STEPS: Sequence[StepBase] = [
    step1, step2, step3, *steps4, step5, step6, step7, step8
]

futures: dict[str, PrefectFuture[FSEnvironment]
              ] = BFEDriver.run_steps(STEPS, RES_STORE)
final_env: FSEnvironment = futures["Download the best boulderNet model"].result(
)
