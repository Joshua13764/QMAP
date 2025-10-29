from pathlib import Path
import attr
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from typing import Any, Dict

from bennu_feature_extractor.environment_tools.base_classes.file_storage_persist_base import FileStoragePersistBase

@attr.define(slots=True)
class MemoryBlock():
    path : Path
    persist : FileStoragePersistBase
    blob : Any

class RuntimeMemory(FileStorageMediumBase):
    def __init__(self) -> None:
        self.memory : Dict[str, MemoryBlock] = {} # virtual path : memory block

    def save(self, obj: Any, virtual_path : Path, persist : FileStoragePersistBase) -> None:

        self.memory[virtual_path.as_uri()] = MemoryBlock(
            path = virtual_path,
            persist = persist,
            blob = obj
        )

    def load(self, virtual_path : Path) -> Any:
        return self.memory[virtual_path.as_uri()]
    
    def does_path_exist(self, virtual_path : Path) -> bool:
        return virtual_path.as_uri() in self.memory
