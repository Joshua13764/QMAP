from typing import Any

import tifffile
from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from numpy.typing import NDArray


class FSTiffAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return tifffile.imread(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        raise tifffile.imwrite(path.actual_path.as_posix(), obj)
