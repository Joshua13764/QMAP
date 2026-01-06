import shutil
from dataclasses import dataclass
from os import remove

from attr import field

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSShutilCopyAdapter(FSAdapterBase[FSPathLocalDisk, FSPathLocalDisk]):
    """Copies a file from the src (object) to a path (path)"""
    overwrite: bool = field(default=True)

    def read(self, path: FSPathLocalDisk) -> FSPathLocalDisk:
        raise NotImplementedError

    def write(self, obj: FSPathLocalDisk, path: FSPathLocalDisk) -> None:
        if path.exists and self.overwrite:
            remove(path.actual_path)
        elif path.exists:
            raise FileExistsError

        shutil.copy2(
            src=obj.actual_path,
            dst=path.actual_path
        )
