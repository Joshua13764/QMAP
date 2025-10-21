from pathlib import Path

from bennu_feature_extractor.logger_factory import get_logger

from bennu_feature_extractor_BoulderNet.Best_model_downloader import \
    BestModelDownloader

logger = get_logger("test_integration", log_dir=Path("./logs"))

stepA = BestModelDownloader(
    logger,
    Path("./downloads").as_posix(),
    Url="https://zenodo.org/records/8171052/files/best_model.zip?download=1").run()
