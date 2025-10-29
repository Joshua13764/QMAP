from typing import List
import attr
from pathlib import Path
from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FileStorageAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from bennu_feature_extractor.environment_tools.file_storage_adapters.general_adapter import GeneralAdapter
from bennu_feature_extractor.environment_tools.file_storage_adapters.pds4_adapter import PDS4Adapter
from bennu_feature_extractor.environment_tools.file_storage_adapters.pickle_adapter import PickleAdapter
from bennu_feature_extractor.environment_tools.file_storage_adapters.png_adapter import PNGAdapter
from bennu_feature_extractor.environment_tools.file_storage_adapters.txt_adapter import TXTAdapter


@attr.define()
class FileStorageEnvironment():
    mediums : List[FileStorageMediumBase]
    adapters : List[FileStorageAdapterBase] = [
        TXTAdapter(), PNGAdapter(), PickleAdapter(), PDS4Adapter(), GeneralAdapter()
    ]

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