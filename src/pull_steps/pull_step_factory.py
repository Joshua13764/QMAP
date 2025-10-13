from .steps.pds import PDSPullStep
from .steps.BoulderNet_best_model import BoulderNetBestModelLoadStep

from logging import Logger

class PullStepFactory:
    @staticmethod
    def create_PDS_pull_pipeline_step(
        logger : Logger,
        path : str,
        url : str) -> PDSPullStep:
        return PDSPullStep(
            DownloadPath = path,
            Url = url,
            _logger = logger
        )
    
    @staticmethod
    def create_BoulderNetBestModel_pull_pipeline_step(
        logger : Logger,
        path : str,
        url : str) -> BoulderNetBestModelLoadStep:
        return BoulderNetBestModelLoadStep(
            DownloadPath = path,
            Url = url,
            _logger = logger
        )