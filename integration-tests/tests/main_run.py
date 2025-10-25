from pathlib import Path

from bennu_feature_extractor.environment import Environment
from bennu_feature_extractor.logger_factory import get_logger
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader
from prefect import flow

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


@flow()
def data_loader_flow() -> None:

    env: Environment = Environment(log_path=Path("./logs"))

    stepA: StepBase = BestModelDownloader(
        logger,
        dataDownloadPath.as_posix(),
        Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1"
    )

    env = stepA.run(env)

    for url in urls_to_download:
        logger.info(f"Downloading data from {url}")

        step: StepBase = PDSDownloader(
            logger,
            dataDownloadPath.as_posix(),
            Url=url
        )

        env = step.run(env)


data_loader_flow()
