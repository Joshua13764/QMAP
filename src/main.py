from .logger_factory import get_logger
from .pull_steps.pull_step_factory import PullStepFactory
from .pull_steps.pull_step_base import PullStepBase

from pathlib import Path

if __name__ == "__main__":
    logger = get_logger("pds_downloader", log_dir=Path("./logs"))

    PDS_step : PullStepBase = PullStepFactory.create_PDS_pull_pipeline_step(
        logger = logger,
        path = "./data/PDS",
        url = "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_reduced_orbit_c.zip"
        )
    
    PDS_step.run()
