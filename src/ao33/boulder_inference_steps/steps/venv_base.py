from ..boulder_inference_step_base import BoulderInferenceStepBase

from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod, ABC
import subprocess
from jsondataclasses import jsondataclass
import json

@jsondataclass
class ServerConfig:
    host: StopAsyncIteration
    port: int

class IVenvClient(ABC):
    pass

@dataclass
class venvBase(BoulderInferenceStepBase):
    VenvPath : Path
    ServerPath : Path
    server_handle = None
    server_config : ServerConfig

    @abstractmethod
    def setup_venv(self):
        pass

    @abstractmethod
    def get_venv_client(self) -> IVenvClient:
        pass

    def start_venv_server(self) -> None:
        if self.server_handle and self.server_handle.poll() is None:
            self._logger.warning("Server is already running.")
            return

        venv_python_path = self.VenvPath / "venv" / "Scripts" / "python.exe"
        self._logger.info(f"Starting server {self.ServerPath} in venv {self.VenvPath}")

        self.server_config = ServerConfig(
            host = "127.0.0.1",
            port = 5000
        )

        self._logger.info(f"Using server config: {ServerConfig}")

        with open(self.ServerPath / "server_config.json", "w", encoding="utf-8") as fh:
            json.dump({"host": self.server_config, "port": self.server_config}, fh, indent=2)

        self.server_handle = subprocess.Popen(
            [str(venv_python_path), "server.py"],
            cwd=self.ServerPath,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        self._logger.info(f"Server started with PID: {self.server_handle.pid}")

    def stop_venv_server(self) -> None:
        
        if self.server_handle and self.server_handle.poll() is None:
            self._logger.info(f"Stopping server PID: {self.server_handle.pid}...")

            self.server_handle.terminate()

            try:

                self.server_handle.wait(timeout=5)
                self._logger.info("Server terminated gracefully")

            except subprocess.TimeoutExpired:
                self._logger.warning("Graceful stop timed out. Forcing stop...")
                self.server_handle.kill()
                self._logger.info("Server forcefully stopped")
        else:
            self._logger.info("No running server process to stop.")

    def does_venv_exist(self) -> bool:
        return self.VenvPath.exists() and (self.VenvPath / "Scripts" / "python.exe").exists()

    def run_shell_command(self, command: str, cwd : Path) -> str:

        self.create_path(cwd)
        self._logger.info(f"Running shell command: {command}")

        try:
            result = subprocess.run(
                command, 
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

