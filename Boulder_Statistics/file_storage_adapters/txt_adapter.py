from dataclasses import dataclass

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSTXTAdapter(FSAdapterBase[str, FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> str:
        with path.actual_path.open("rb") as f:
            return f.read().decode('utf-8')

    def write(self, obj: str, path: FSPathLocalDisk) -> None:
        path.make_directory()
        with path.actual_path.open("wb") as f:
            f.write(obj.__repr__().encode('utf-8'))
