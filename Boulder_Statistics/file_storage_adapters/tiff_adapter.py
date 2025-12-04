from dataclasses import dataclass
from typing import Any

import tifffile
from numpy.typing import NDArray

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSTiffAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):
    """Uses the tifffile module to read / write images with a wide range of precision (including float 64)"""

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return tifffile.imread(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        return tifffile.imwrite(path.actual_path.as_posix(), obj)
