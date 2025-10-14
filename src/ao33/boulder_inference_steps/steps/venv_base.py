from ..boulder_inference_step_base import BoulderInferenceStepBase
from .I_venv_client import IVenvClient

from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod
import subprocess
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class ServerConfig:
    host: str
    port: int

@dataclass
class venvBase(BoulderInferenceStepBase):
    VenvPath : Path
    ServerPath : Path
    server_config : ServerConfig

    _server_handle = None

    @abstractmethod
    def setup_venv(self):
        pass

    @abstractmethod
    def get_venv_client(self) -> IVenvClient:
        pass

    def start_venv_server(self) -> None:
        if self._server_handle and self._server_handle.poll() is None:
            self._logger.warning("Server is already running.")
            return

        venv_python_path = self.VenvPath / "venv" / "Scripts" / "python.exe"
        self._logger.info(f"Starting server {self.ServerPath} in venv {self.VenvPath}")

        self._logger.info(f"Using server config: {ServerConfig}")

        with open(self.ServerPath.parent / "server_config.json", "w", encoding="utf-8") as fh:
            fh.write(self.server_config.to_json())

        self._server_handle = subprocess.Popen(
            [str(venv_python_path), str(self.ServerPath.name)],
            cwd=self.ServerPath.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        self._logger.info(f"Server started with PID: {self._server_handle.pid}")

    def stop_venv_server(self) -> None:
        
        if self._server_handle and self._server_handle.poll() is None:
            self._logger.info(f"Stopping server PID: {self._server_handle.pid}...")

            self._server_handle.terminate()

            try:

                self._server_handle.wait(timeout=5)
                self._logger.info("Server terminated gracefully")

            except subprocess.TimeoutExpired:
                self._logger.warning("Graceful stop timed out. Forcing stop...")
                self._server_handle.kill()
                self._logger.info("Server forcefully stopped")
        else:
            self._logger.info("No running server process to stop.")

    def does_venv_exist(self) -> bool:
        return self.VenvPath.exists() and (self.VenvPath / "Scripts" / "python.exe").exists()

    def run_shell_command(self, command: str, cwd : Path) -> str:

        self.create_path(cwd)
        self._logger.info(f"Running shell command: {command}")

        full_cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            command
        ]

        try:
            result = subprocess.run(
                full_cmd, 
                shell = True, 
                check = True,
                text = True, 
                capture_output = True,
                cwd = cwd
            )

            self._logger.info(result.stdout.strip())

            if result.stderr:
                self._logger.error(result.stderr.strip())

        except subprocess.CalledProcessError as e:
            self._logger.error(f"Command failed with error : {e.stderr}")
            raise

        return result.stdout.strip()
    
    def create_path(self, path : Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

