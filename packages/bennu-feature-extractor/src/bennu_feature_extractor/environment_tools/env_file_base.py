from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
import attrs
from typing import Optional, Any
from prefect import get_run_logger

@attrs.define(frozen=True, slots=True, cache_hash=True)
class EnvFileBase(ABC):
    last_modified : Optional[float]
    actual_path_str : str
    virtual_path_str : str

    @property
    def actual_path(self) -> Path:
        return Path(self.actual_path_str)

    @property
    def virtual_path(self) -> Path:
        return Path(self.virtual_path_str)

    @property
    def logger(self) -> Logger:
        return get_run_logger()

    @property
    def file_type(self) -> type:
        return type(self)

    @abstractmethod
    def read(self) -> Any:
        raise NotImplementedError()
    
    @abstractmethod
    def write(self, data: Any) -> None:
        raise NotImplementedError()


    def exists(self) -> bool:
        return self.actual_path.exists()
    
    def delete(self) -> None:
        if self.exists():
            self.actual_path.unlink()
            self.logger.info(f"Deleted file at {self.actual_path}")
        else:
            self.logger.warning(f"Attempted to delete non-existent file at {self.actual_path}")

    def get_size(self) -> int:
        if self.exists():
            return self.actual_path.stat().st_size
        else:
            raise FileExistsError(f"File at {self.actual_path} does not exist. Size is 0.")

    def get_last_modified(self) -> float:
        if self.exists():
            self.last_modified = self.actual_path.stat().st_mtime
            return self.last_modified
        else:
            raise FileExistsError(f"File at {self.actual_path} does not exist. Cannot get last modified time.")
        
    def check_metadata_valid(self) -> bool:
        if not self.last_modified:
            self.logger.warning("Last modified timestamp is not set so failed check_metadata_valid.")
            return False

        if not self.exists():
            return False
        
        current_last_modified = self.get_last_modified()
        return current_last_modified == self.last_modified