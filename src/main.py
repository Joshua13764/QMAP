from .logger_factory import get_logger
from .pull_steps.pull_step_factory import PullStepFactory
from .pull_steps.pull_step_base import PullStepBase

from pathlib import Path
from logging import Logger
from joblib import Parallel, delayed

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
        path=str(base_path),
        url=url
    )
    step.run()

if __name__ == "__main__":
    logger = get_logger("pds_downloader", log_dir=Path("./logs"))
    base_path = Path("./data/PDS")

    # Download all datasets
    [download_pds_dataset(url, logger, base_path) for url in coreDownloads]
