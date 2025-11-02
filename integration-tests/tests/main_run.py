import warnings
from pathlib import Path

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from bennu_feature_extractor_PDS.PDS_to_PNG import PDS_to_PNG
from bennu_feature_extractor_PDS.SPICE_kernels_downloader import \
    SPICEKernelGrabber
from graphviz import Source
from prefect import flow
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner

warnings.filterwarnings(
    "ignore",
    message="Config key `toml_file` is set in model_config"
)

# # To be run once
# run_dir_store = LocalFileSystem(basepath=".\\.run_dir_storage")
# run_dir_store.save("run-dir-storage", overwrite=True)

run_dir_store = LocalFileSystem.load("run-dir-storage")

model_download_path: Path = Path(r"F:\AO33\AO33_models")
pds_download_path: Path = Path(r"F:\AO33\AO33_pds_DATA")
pipeline_working_path: Path = Path(r"F:\AO33\AO33_pipeline_DATA")
spice_download_path: Path = Path(r"F:\AO33\AO33_SPICE_DATA")


urls_to_download = [
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

    tasks.append(
        BestModelDownloader(
            run_dir_store,
            model_download_path.as_posix(),
            Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
        ).get_task_no_cache.submit(FSEnvironment.empty())
    )

    tasks += [
        PDSDownloader(
            run_dir_store,
            pds_download_path.as_posix(),
            Url=url
        ).get_task_no_cache.submit(FSEnvironment.empty())
        for url in urls_to_download
    ]

    wait(tasks)
    envs: list[FSEnvironment] = [f.result() for f in tasks]

    returned_env: FSEnvironment = FSEnvironment.merge(envs)

    # labels = ["BestModelDownloader: best_model.zip"] + [
    #     f"PDSDownloader: {Path(u).name}" for u in urls_to_download
    # ]
    # dot = [
    #     "digraph data_loader {",
    #     "  rankdir=LR;",
    #     '  node [shape=box, style=rounded];',
    #     *(f'  "{name}";' for name in labels),
    #     '  "merge_environments" [shape=ellipse];',
    #     *(f'  "{name}" -> "merge_environments";' for name in labels),
    #     "}"
    # ]
    # Source(
    #     "\n".join(dot)).render(
    #     filename="flow_graph",
    #     format="svg",
    #     cleanup=True)
    # logger.info("Wrote flow_graph.png")

    return returned_env


@flow()
def data_convert_flow(env: FSEnvironment) -> FSEnvironment:
    pds_to_png_task = PDS_to_PNG(
        result_storage=run_dir_store,
        cluster_key="ocams_data_calibrated_detailed_survey",
        run_path=pipeline_working_path
    ).get_task_no_cache.submit(env)

    converted_env = pds_to_png_task.result()
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
    ).get_task_no_cache.submit(FSEnvironment.empty())

    return fut.result()


if __name__ == "__main__":
    # env = data_loader_flow()
    # data_convert_flow(env)
    spice_kernals_loader_flow()
