from pathlib import Path

from bennu_feature_extractor.environment import Environment
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from bennu_feature_extractor_PDS.PDS_to_PNG import PDS_to_PNG
from graphviz import Source
from prefect import flow
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner

# # To be run once
# run_dir_store = LocalFileSystem(basepath=".\\.run_dir_storage")
# run_dir_store.save("run-dir-storage", overwrite=True)

run_dir_store = LocalFileSystem.load("run-dir-storage")
dataDownloadPath = Path("C:\\Users\\Joshu\\Documents\\AO33_DATA")

urls_to_download = [
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_reduced_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_orbit_b.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_recon.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_metadata.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_calibration.zip"
]


@flow(task_runner=ThreadPoolTaskRunner(max_workers=20))
def data_loader_flow() -> Environment:
    tasks: list[PrefectFuture[Environment]] = []

    tasks.append(
        BestModelDownloader(
            run_dir_store,
            dataDownloadPath.as_posix(),
            Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
        ).get_task.submit(Environment())
    )

    tasks += [
        PDSDownloader(
            run_dir_store,
            dataDownloadPath.as_posix(),
            Url=url
        ).get_task.submit(Environment())
        for url in urls_to_download
    ]

    wait(tasks)
    envs: list[Environment] = [f.result() for f in tasks]

    returned_env: Environment = Environment.merge_environments(envs)

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
def data_convert_flow(env: Environment) -> Environment:
    pds_to_png_task = PDS_to_PNG(
        result_storage=run_dir_store,
        cluster_key="ocams_data_calibrated_detailed_survey",
        run_path=Path(r"F:\AO33_DATA2")
    ).get_task.submit(env)

    converted_env = pds_to_png_task.result()
    return converted_env


if __name__ == "__main__":
    env = data_loader_flow()
    data_convert_flow(env)
