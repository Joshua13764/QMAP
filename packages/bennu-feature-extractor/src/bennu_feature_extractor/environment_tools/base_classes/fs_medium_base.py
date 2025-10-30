from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path

import attr
from prefect import get_run_logger


@attr.define()
class FileStorageMediumBase(ABC):
    name: str

    @abstractmethod
    def does_path_exist(self, virtual_path: Path) -> bool:
        ...

    @property
    def _logger(self) -> Logger:
        return get_run_logger()
