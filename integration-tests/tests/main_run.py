from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Coroutine, FrozenSet, Iterable, List, Mapping, Sequence

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
from prefect import flow, task
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner

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

STEPS: Sequence[StepBase] = [
    BestModelDownloader(
        task_name="Download the best boulderNet model",
        run_after_task_names=frozenset(),
        DownloadPath=model_download_path.as_posix(),
        Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
    ),

    SimpleRequest(
        task_name=f"Downloader for the bennu PAN file",
        run_after_task_names=frozenset(),
        url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit.tif",
        fs_path=pds_download_path.as_posix(),
        sub_path=Path("OCAMS", "Global PAN Mosaic.tif").as_posix(),
        markers=frozenset([FSMarkerString(value="PAN_texture")])
    ),

    SimpleRequest(
        task_name=f"Downloader for the bennu OBJ (LQ) mesh",
        run_after_task_names=frozenset(),
        url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/g_00880mm_alt_ptm_0000n00000_v020.obj",
        fs_path=pds_download_path.as_posix(),
        sub_path=Path(
            "OCAMS",
            "Global Bennu 3D model - OLA v20 PTM.obj").as_posix(),
        markers=frozenset(
            [FSMarkerString(value="OCAMS Model"), FSMarkerString("ProjectModel")])
    ),

    *(PDSDownloader(
        task_name=f"Downloader for PDS file {url}",
        run_after_task_names=frozenset(),
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
    )
]


@task(name="merge_envs")
def merge_envs(
    base_env: "FSEnvironment",
    upstream_envs: list["FSEnvironment"],
) -> "FSEnvironment":
    return FSEnvironment.merge(
        upstream_envs +
        [base_env])


@flow(name="run_step_dag")
def run_step_dag(
        steps: list[StepBase], result_storage) -> dict[str, PrefectFuture[FSEnvironment]]:
    base_env: FSEnvironment = FSEnvironment.empty()
    name_to_step: dict[str, StepBase] = {s.task_name: s for s in steps}
    graph: dict[str, set[str]] = {s.task_name: set(
        s.run_after_task_names) for s in steps}
    ts: TopologicalSorter[str] = TopologicalSorter(graph)
    order: List[str] = list(ts.static_order())

    futures: dict[str, PrefectFuture[FSEnvironment]] = {}

    for task_name in order:
        step: StepBase = name_to_step[task_name]
        upstream_names: FrozenSet[str] = step.run_after_task_names
        upstream_futures: List[PrefectFuture[FSEnvironment]] = [
            futures[n] for n in upstream_names]

        if upstream_futures:
            # ⬇️ Prefect 2 style: no wait_for here
            merged_env_future: PrefectFuture[FSEnvironment] = merge_envs.submit(
                base_env,
                upstream_futures,   # list of PrefectFuture[FSEnvironment]
            )
            env_arg: PrefectFuture[FSEnvironment] = merged_env_future
        else:
            env_arg = base_env

        compiled_task = step.get_task(result_storage)

        # ⬇️ Again: no wait_for in Prefect 2
        future: PrefectFuture[FSEnvironment] = compiled_task.submit(
            env_arg,
            step,  # your ArchiveDownloadBase or other subclass
        )

        futures[task_name] = future

    return futures


futures: dict[str, PrefectFuture[FSEnvironment]
              ] = run_step_dag(STEPS, RES_STORE)
final_env: FSEnvironment = futures["Download the best boulderNet model"].result(
)


# @flow()
# def data_convert_flow(env: FSEnvironment) -> FSEnvironment:
#     pds_to_png_task: PrefectFuture[FSEnvironment] = PDS_to_PNG(
#         result_storage=run_dir_store,
#         cluster_key="ocams_data_calibrated_detailed_survey",
#         run_path=pipeline_working_path
#     ).submit_task(env)

#     converted_env: FSEnvironment = pds_to_png_task.result()
#     return converted_env


# @flow(task_runner=ThreadPoolTaskRunner(max_workers=20))
# def spice_kernals_loader_flow() -> FSEnvironment:
#     # One task that mirrors everything referenced by the MK(s)
#     fut: PrefectFuture[FSEnvironment] = SPICEKernelGrabber(
#         result_storage=run_dir_store,
#         DownloadPath=spice_download_path.as_posix(),
#         MkUrls=MK_URLS,
#         ExtraUrls=EXTRA_URLS,
#     ).submit_task()

#     return fut.result()


# @flow()
# def pp_tasks_flow(env: FSEnvironment) -> FSEnvironment:
#     pan_env = PANToLOD(
#         result_storage=run_dir_store,
#         root_path=pipeline_working_path,
#         lod_res=1024,
#         skip_if_exists=True
#     ).submit_task(env).result()

#     # OBJToLAS(
#     #     result_storage=run_dir_store,
#     #     root_path=pipeline_working_path,
#     #     lod_res=1024,
#     #     depth=4,
#     #     skip_if_exists=False,
#     #     debug_mode=True
#     # ).submit_task(env).result()

#     PDS4BoulderNetInference(
#         result_storage=run_dir_store,
#         run_path=pipeline_working_path_fast
#     ).submit_task(pan_env).result()

#     return pan_env


# if __name__ == "__main__":
#     env: FSEnvironment = data_loader_flow()
#     # spice_kernals_loader_flow()
#     env2: FSEnvironment = data_convert_flow(env)
#     pp_tasks_flow(env2)
