from dataclasses import dataclass

import trimesh

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSTrimeshAdapter(FSAdapterBase[trimesh.Trimesh, FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> trimesh.Trimesh:
        return trimesh.load_mesh(
            path.actual_path.as_posix(), file_type="obj", process=True)

    def write(self, obj: trimesh.Trimesh, path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
