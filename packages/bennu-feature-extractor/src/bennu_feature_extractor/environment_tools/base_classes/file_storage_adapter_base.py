from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_persist_base import FileStoragePersistBase
from bennu_feature_extractor.environment_tools.file_storage_environment import FileStorageEnvironment

class FileStorageAdapterBase(ABC):
    @abstractmethod
    def save(self, obj: Any, virtual_path : Path, persist : FileStoragePersistBase, medium : FileStorageMediumBase) -> None:
        ...

    @abstractmethod
    def load(self, virtual_path: Path, env : FileStorageEnvironment) -> Any:
        ...