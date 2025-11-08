from typing import Any

import trimesh
from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


class FSTrimeshAdapter(
        FSAdapterBase[Any, FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> Any:
        return trimesh.load(path.actual_path.as_posix(),
                            force='mesh', process=True)

    def write(self, obj: Any, path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
