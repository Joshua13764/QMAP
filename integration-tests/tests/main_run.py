from pathlib import Path
from typing import Any, Coroutine

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from bennu_feature_extractor.step_templates.simple_request import SimpleRequest
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_PDS.OBJ_to_LAS import OBJToLAS
from bennu_feature_extractor_PDS.PAN_to_LOD import PANToLOD
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from bennu_feature_extractor_PDS.PDS_to_PNG import PDS_to_PNG
from bennu_feature_extractor_PDS.SPICE_kernels_downloader import \
    SPICEKernelGrabber
from prefect import flow
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner

# # To be run once
# run_dir_store = LocalFileSystem(basepath=".\\.run_dir_storage")
# run_dir_store.save("run-dir-storage", overwrite=True)

run_dir_store: LocalFileSystem | Coroutine[Any, Any,
                                           LocalFileSystem] = LocalFileSystem.load("run-dir-storage")

model_download_path: Path = Path(r"F:\AO33\AO33_models")
pds_download_path: Path = Path(r"F:\AO33\AO33_pds_DATA")
pipeline_working_path: Path = Path(r"F:\AO33\AO33_pipeline_DATA")
spice_download_path: Path = Path(r"F:\AO33\AO33_SPICE_DATA")


urls_to_download: list[str] = [
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_reduced_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_orbit_b.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_recon.zip",
    # "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_metadata.zip",
    # "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_calibration.zip"
]


@flow(task_runner=ThreadPoolTaskRunner(max_workers=20))
def data_loader_flow() -> FSEnvironment:
    tasks: list[PrefectFuture[FSEnvironment]] = []

    # tasks.append(
    #     BestModelDownloader(
    #         run_dir_store,
    #         model_download_path.as_posix(),
    #         Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
    #     ).submit_task()
    # )

    # tasks += [
    #     PDSDownloader(
    #         run_dir_store,
    #         pds_download_path.as_posix(),
    #         Url=url
    #     ).submit_task()
    #     for url in urls_to_download
    # ]

    tasks.append(
        SimpleRequest(
            run_dir_store,
            url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
            fs_path=pds_download_path,
            sub_path=Path("OCAMS", "Global PAN Mosaic.tif"),
            markers=frozenset([FSMarkerString(value="PAN_texture")])
        ).submit_task()
    )

    tasks.append(
        SimpleRequest(
            run_dir_store,
            url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/g_00880mm_alt_ptm_0000n00000_v020.obj",
            fs_path=pds_download_path,
            sub_path=Path("OCAMS", "Global Bennu 3D model - OLA v20 PTM.obj"),
            markers=frozenset(
                [FSMarkerString(value="OCAMS Model"), FSMarkerString("ProjectModel")])
        ).submit_task()
    )

    wait(tasks)
    envs: list[FSEnvironment] = [f.result() for f in tasks]

    returned_env: FSEnvironment = FSEnvironment.merge(envs)

    return returned_env


@flow()
def data_convert_flow(env: FSEnvironment) -> FSEnvironment:
    pds_to_png_task: PrefectFuture[FSEnvironment] = PDS_to_PNG(
        result_storage=run_dir_store,
        cluster_key="ocams_data_calibrated_detailed_survey",
        run_path=pipeline_working_path
    ).submit_task(env)

    converted_env: FSEnvironment = pds_to_png_task.result()
    return converted_env


# Published 2019 metakernel (covers Detailed Survey, Orbit B, Recon in 2019)
MK_URLS = [
    "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/mk/orx_2019_v08.tm",
]

# Optional: add a Bennu DSK as a plain file (no extraction)
EXTRA_URLS = [
    # "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/dsk/bennu_g_00880mm_alt_obj_0000n00000_v021a.bds"
]


@flow(task_runner=ThreadPoolTaskRunner(max_workers=20))
def spice_kernals_loader_flow() -> FSEnvironment:
    # One task that mirrors everything referenced by the MK(s)
    fut: PrefectFuture[FSEnvironment] = SPICEKernelGrabber(
        result_storage=run_dir_store,
        DownloadPath=spice_download_path.as_posix(),
        MkUrls=MK_URLS,
        ExtraUrls=EXTRA_URLS,
    ).submit_task()

    return fut.result()


@flow()
def pp_tasks_flow(env: FSEnvironment) -> FSEnvironment:
    # PANToLOD(
    #     result_storage=run_dir_store,
    #     root_path=pipeline_working_path,
    #     lod_res=1024,
    #     skip_if_exists=True
    # ).submit_task(env).result()

    # OBJToLAS(
    #     result_storage=run_dir_store,
    #     root_path=pipeline_working_path,
    #     lod_res=1024,
    #     depth=4,
    #     skip_if_exists=False,
    #     debug_mode=True
    # ).submit_task(env).result()

    return None


if __name__ == "__main__":
    env: FSEnvironment = data_loader_flow()
    pp_tasks_flow(env)
    # data_convert_flow(env)
    # spice_kernals_loader_flow()
