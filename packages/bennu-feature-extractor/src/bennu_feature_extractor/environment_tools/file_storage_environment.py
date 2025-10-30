from typing import List
import attr
from pathlib import Path
from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FileStorageAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from bennu_feature_extractor.environment_tools.file_storage_persists.runtime_only_persist import RuntimeOnlyPersist

@attr.define()
class FileStorageEnvironment():
    mediums : List[FileStorageMediumBase]

    def get_medium(self, virtual_path : Path) -> FileStorageMediumBase:
        matches : List[FileStorageMediumBase] = [medium for medium in self.mediums if medium.does_path_exist(virtual_path)]

        if len(matches) == 0: raise FileNotFoundError(f"The file in path {virtual_path} cannot be found")
        if len(matches) >= 2: raise FileNotFoundError(f"The file in path {virtual_path} occupies multiple mediums {[match.name for match in matches]}")

        return matches[0]
    
    def get_medium_by_name(self, name : str):
        matches : List[FileStorageMediumBase] = [medium for medium in self.mediums if medium.name == name]

        if len(matches) == 0: raise FileNotFoundError(f"The medium {name} cannot be found")
        if len(matches) >= 2: raise FileNotFoundError(f"The medium {name} occurs multiple times {[match.name for match in matches]}")

        return matches[0]
    
    def save[T](self, medium_name : str, obj : T, virtual_path : Path, adapter : FileStorageAdapterBase[T]) -> None:
        adapter.save(obj, virtual_path, RuntimeOnlyPersist(), self.get_medium_by_name(medium_name))

    def load[T](self, virtual_path : Path, adapter : FileStorageAdapterBase[T]) -> T:
        return adapter.load(virtual_path, self)

