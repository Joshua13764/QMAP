from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSNumpyAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):
    """Uses the np module to read / write arrays"""

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return np.load(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        return np.save(path.actual_path.as_posix(), obj)
