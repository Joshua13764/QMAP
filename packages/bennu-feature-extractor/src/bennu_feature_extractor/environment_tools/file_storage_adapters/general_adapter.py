from typing import Any
from pathlib import Path
from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FileStorageAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_persist_base import FileStoragePersistBase
from bennu_feature_extractor.environment_tools.file_storage_environment import FileStorageEnvironment
from bennu_feature_extractor.environment_tools.file_storage_mediums.runtime_memory import RuntimeMemory

class GeneralAdapter(FileStorageAdapterBase[Any]):

    def save(self, obj: Any, virtual_path: Path, persist: FileStoragePersistBase, medium: FileStorageMediumBase) -> None:
        match medium:
            case RuntimeMemory() as rm:
                rm.save(obj, virtual_path, persist)

            case _:
                raise NotImplementedError(f"The adapter {type(self).__name__} has no method to handel saving to the medium {type(medium).__name__}")
                
    
    def load(self, virtual_path: Path, env : FileStorageEnvironment) -> Any:
        match env.get_medium(virtual_path):
            case RuntimeMemory() as rm:
                return rm.load(virtual_path)

            case _ as medium:
                raise NotImplementedError(f"The adapter {type(self).__name__} has no method to handel loading to the medium {type(medium).__name__}")