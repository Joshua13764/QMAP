import os
import shutil

from bennu_feature_extractor.environment import Environment
from bennu_feature_extractor.step_templates.venv.venv_base import venvBase
from bennu_feature_extractor.step_templates.venv.venv_client_base import \
    VenvClientBase

from .BoulderNet_venv_client import BoulderNetVenvClient


class BoulderNetDefault(venvBase):
    @property
    def name(self) -> str:
        return __name__

    def run(self, env: Environment) -> Environment:
        raise NotImplementedError

    def setup_venv(self):

        if os.path.exists(self.VenvPath):
            shutil.rmtree(self.VenvPath)
            os.makedirs(self.VenvPath)

        # Create venv
        self.run_shell_command("py -3.10 -m venv venv", self.VenvPath)

        # Update pip
        self.run_pip_shell_command("install --upgrade pip", self.VenvPath)

        # Install Rastertools
        self.run_shell_command(
            "git clone https://github.com/astroNils/rastertools.git",
            self.VenvPath)
        self.run_pip_shell_command(
            "install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING",
            self.VenvPath / "rastertools"
        )
        self.run_pip_shell_command(
            "install -r requirements.txt",
            self.VenvPath / "rastertools")

        # Install Shptools
        self.run_shell_command(
            "git clone https://github.com/astroNils/shptools.git",
            self.VenvPath)
        self.run_pip_shell_command(
            "install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING",
            self.VenvPath / "shptools"
        )
        self.run_pip_shell_command(
            "install -r requirements.txt",
            self.VenvPath / "shptools")

        # Install Pytorch (CPU)
        self.run_pip_shell_command(
            "install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 "
            "--extra-index-url https://download.pytorch.org/whl/cpu",
            self.VenvPath
        )

        # Install Detectron2 (CPU)
        self.run_pip_shell_command(
            'install --no-build-isolation --extra-index-url '
            'https://download.pytorch.org/whl/cpu "git+https://github.com/facebookresearch/detectron2.git"',
            self.VenvPath
        )

        # Install MLtools
        self.run_shell_command(
            "git clone https://github.com/astroNils/MLtools.git",
            self.VenvPath)
        self.run_pip_shell_command(
            "install --index-url https://test.pypi.org/simple/ --no-deps MLtools_BOULDERING",
            self.VenvPath / "MLtools"
        )
        self.run_pip_shell_command(
            "install -r requirements.txt",
            self.VenvPath / "MLtools")

        # Install other dependencies (for server use)
        self.run_pip_shell_command(
            "install fastapi uvicorn pydantic dataclasses_json requests rich",
            self.VenvPath)

    def validate_venv(self) -> bool:
        return True

    def get_venv_client(self) -> VenvClientBase:
        # self.start_venv_server()
        return BoulderNetVenvClient(self.logger, self.server_config)
