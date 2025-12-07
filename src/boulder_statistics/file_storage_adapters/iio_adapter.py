from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v3 as iio
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSIIOAdapter(
        FSAdapterBase[NDArray[Any], FSPathLocalDisk]):

    add_file_extension: str | None = field(default=None)

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return iio.imread(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        save_path: Path = path.actual_path

        if self.add_file_extension is not None:
            save_path = save_path.with_suffix(
                self.add_file_extension)

        return iio.imwrite(save_path, obj)
