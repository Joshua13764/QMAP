from pathlib import Path

from bennu_feature_extractor.logger_factory import get_logger
from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader

logger = get_logger("test_PDS_dowloads_required_data", log_dir=Path("./logs"))

urls_to_download = [
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_reduced_detailed_survey.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_orbit_b.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_data_calibrated_recon.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_metadata.zip",
    "https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_calibration.zip"
    ]

stepA = PDSDownloader(logger, Path("./downloads").as_posix(), Url="https://sbnarchive.psi.edu/pds4/orex/downloads_ocams/ocams_calibration.zip").run()
