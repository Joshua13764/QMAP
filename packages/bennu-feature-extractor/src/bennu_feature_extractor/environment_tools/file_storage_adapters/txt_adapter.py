from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import FSPathLocalDisk

class FSTXTAdapter(FSAdapterBase[str, FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> str:
        with path.actual_path.open("rb") as f:
            return f.read().decode('utf-8')
        
    def write(self, obj: str, path: FSPathLocalDisk) -> None:
        with path.actual_path.open("wb") as f:
            f.write(obj.__repr__().encode('utf-8'))