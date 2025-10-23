from abc import ABC, abstractmethod
from logging import Logger, getLogger
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvFileBase(ABC):
    last_modified : Optional[float]
    actual_path : Path
    virtual_path : Path

    # Logger is excluded from serialization
    logger: Logger = field(
        default_factory=lambda: getLogger(__name__),
        repr=False,
        compare=False,
    )

    # Pickle support
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = getLogger(__name__)
    
    @abstractmethod
    def read(self) -> object:
        raise NotImplementedError()
    
    @abstractmethod
    def write(self, data: object) -> None:
        raise NotImplementedError()
    
    @property
    def file_type(self) -> type:
        return type(self)
    
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