from .venv_base import venvBase, IVenvClient

from dataclasses import dataclass
from pathlib import Path
import requests

class BoulderNetDefault(venvBase):
    @property
    def name(self) -> str:
        return __name__
    
    def run(self):
        print("Dummy load step executed.")

    def setup_venv(self):
        # Create venv
        self.run_shell_command("py -3.10 -m venv venv", self.VenvPath)
        # self.run_shell_command("Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process", self.VenvPath)
        self.run_shell_command(".\\venv\\Scripts\\Activate.ps1", self.VenvPath)

        # # Install Rastertools 
        # self.run_shell_command("git clone https://github.com/astroNils/rastertools.git", self.VenvPath)
        # self.run_shell_command(
        #     "python -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING",
        #     self.VenvPath / "rastertools"
        # )
        # self.run_shell_command("pip install -r requirements.txt", self.VenvPath / "rastertools")

        # # Install Shptools
        # self.run_shell_command("git clone https://github.com/astroNils/shptools.git", self.VenvPath)
        # self.run_shell_command(
        #     "python -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING",
        #     self.VenvPath / "shptools"
        # )
        # self.run_shell_command("pip install -r requirements.txt", self.VenvPath / "shptools")

        # # Install Pytorch (CPU)
        # self.run_shell_command(
        #     "pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 "
        #     "--extra-index-url https://download.pytorch.org/whl/cpu",
        #     self.VenvPath
        # )

        # # Install Detectron2 (CPU)
        # self.run_shell_command(
        #     'pip install --no-build-isolation --extra-index-url '
        #     'https://download.pytorch.org/whl/cpu "git+https://github.com/facebookresearch/detectron2.git"',
        #     self.VenvPath
        # )

        # # Install MLtools
        # self.run_shell_command("git clone https://github.com/astroNils/MLtools.git", self.VenvPath)
        # self.run_shell_command(
        #     "python -m pip install --index-url https://test.pypi.org/simple/ --no-deps MLtools_BOULDERING",
        #     self.VenvPath / "MLtools"
        # )
        # self.run_shell_command("pip install -r requirements.txt", self.VenvPath / "MLtools")

        # Install other dependencies (for server use)
        self.run_shell_command("pip install fastapi uvicorn pydantic dataclasses_json requests", self.VenvPath)

    def validate_venv(self) -> bool:
        return True
    
    def get_venv_client(self) -> IVenvClient:
        return BoulderNetVenvClient(self._logger, self.server_config)
    
class BoulderNetVenvClient(IVenvClient):
    def __init__(self, logger, config):
        self._logger = logger
        self._config = config
        self.base = f"http://{config.host}:{config.port}"
        self._logger.info(f"ServerClient initialized with base URL: {self.base}")

    def test_ping(self):
        r = requests.get(f"{self.base}/ping", timeout=5)
        data = r.json()
        self._logger.info(f"Ping response: {data}")
        return data

    def test_echo(self, text):
        r = requests.post(f"{self.base}/echo", json={"text": text}, timeout=10)
        data = r.json()
        self._logger.info(f"Echo response: {data}")
        return data

    def get_base_url(self):
        return self.base