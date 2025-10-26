from pathlib import Path

from bennu_feature_extractor.environment import Environment
from bennu_feature_extractor.logger_factory import get_logger
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from graphviz import Source
from prefect import flow
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner

logger = get_logger("main_run", log_dir=Path("./logs"))

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

    env: Environment = Environment.get_empty_environment(logger=logger)

    tasks: list[PrefectFuture[Environment]] = []

    tasks.append(
        BestModelDownloader(
            logger,
            dataDownloadPath.as_posix(),
            Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
        ).get_task.submit(Environment.get_empty_environment(logger=logger))
    )

    tasks += [
        PDSDownloader(
            logger,
            dataDownloadPath.as_posix(),
            Url=url
        ).get_task.submit(Environment.get_empty_environment(logger=logger))
        for url in urls_to_download
    ]

    wait(tasks)
    envs: list[Environment] = [f.result() for f in tasks]
    returned_env: Environment = Environment.merge_environments(
        envs, custom_logger=logger)

    labels = ["BestModelDownloader: best_model.zip"] + [
        f"PDSDownloader: {Path(u).name}" for u in urls_to_download
    ]
    dot = [
        "digraph data_loader {",
        "  rankdir=LR;",
        '  node [shape=box, style=rounded];',
        *(f'  "{name}";' for name in labels),
        '  "merge_environments" [shape=ellipse];',
        *(f'  "{name}" -> "merge_environments";' for name in labels),
        "}"
    ]
    Source(
        "\n".join(dot)).render(
        filename="flow_graph",
        format="svg",
        cleanup=True)
    logger.info("Wrote flow_graph.png")

    return returned_env


if __name__ == "__main__":
    data_loader_flow()
