from .logger_factory import get_logger
from .pull_steps.pull_step_factory import PullStepFactory
from .pull_steps.pull_step_base import PullStepBase

from pathlib import Path
from logging import Logger
from joblib import Parallel, delayed
from dataclasses import dataclass

@dataclass
class Paths:
    data_path : Path = Path("./data/PDS")
    models_path : Path = Path("./models/BoulderNet")

# List of OCAMS datasets to download
coreDownloads = [
    # Calibrated Recon
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_recon.zip",
    
    # Calibrated Detailed Survey
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_detailed_survey.zip",
    
    # Metadata
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams.metadata.v9.zip"
]

def download_pds_dataset(url: str, logger: Logger, base_path: Path) -> None:
    """
    Creates a PullStepBase instance for a given PDS URL and runs it.
    """
    step: PullStepBase = PullStepFactory.create_PDS_pull_pipeline_step(
        logger=logger,
        path=base_path.as_posix(),
        url=url
    )
    step.run()

if __name__ == "__main__":
    logger = get_logger("pds_downloader", log_dir=Path("./logs"))
    base_path = Path("./data/PDS")

    # Download all datasets
    [download_pds_dataset(url, logger, Paths.data_path) for url in coreDownloads]

    # Download BoulderNet best model
    PullStepFactory.create_BoulderNetBestModel_pull_pipeline_step(
        logger = logger,
        url = "https://zenodo.org/record/8171052/files/best_model.zip",
        path = Paths.models_path.as_posix()
    ).run()
