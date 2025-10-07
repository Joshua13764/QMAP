from .steps.pds import PDSPullStep

from logging import Logger

class PullStepFactory:
    @staticmethod
    def create_PDS_pull_pipeline_step(logger : Logger, path : str, url : str) -> PDSPullStep:
        return PDSPullStep(
            DownloadPath = path,
            Url = url,
            _logger = logger
        )
