from .steps.BoulderNet_default import BoulderNetDefault
from .steps.venv_base import ServerConfig

from logging import Logger
from pathlib import Path

class BoulderInferenceStepFactory:
    @staticmethod
    def create_BoulderNet_default_BoulderInference_pipeline_step(logger : Logger, venvPath : Path, serverPath : Path) -> BoulderNetDefault:

        server_config : ServerConfig = ServerConfig(host="127.0.0.1", port=5000)

        return BoulderNetDefault(
            _logger = logger,
            VenvPath = venvPath,
            ServerPath = serverPath,
            server_config = server_config
        )
