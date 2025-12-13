from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.img_lod_position import ImgLODPosition

T = TypeVar("T", bound=np.floating)


@dataclass()
class LODImageTile(Generic[T]):
    tile: ImgLODPosition
    get_array_action: Callable[[], NDArray[T]]
    cached_array: NDArray[T] | None = field(default=None)

    @property
    def array(self) -> NDArray[T]:
        if self.cached_array is None:
            self.cached_array = self.get_array_action()

        return self.cached_array

    def get_save_path_from_root_folder(
            self, root_path: FSPathLocalDisk) -> FSPathLocalDisk:

        lod_str_rep: str = self.tile.string_rep
        lod_number: int = self.tile.lod_number

        rel_path: Path = Path(
            f"lod {lod_number}", f"lod tile {lod_str_rep} with shape {
                self.array.shape}"
        )

        root_path.copy_from_folder(rel_path)

        return root_path

    def save(self, folder_path: FSPathLocalDisk,
             adapter: FSAdapterBase) -> None:

        FSEnvironment.save(
            obj=self.array,
            path=self.get_save_path_from_root_folder(folder_path),
            adapter=adapter)

    @classmethod
    def from_file(cls, file_path: FSPathLocalDisk,
                  adapter: FSAdapterBase) -> "LODImageTile[T]":
        lod_str_rep: str = file_path.actual_path.stem.split(" ")[2]

        return cls(
            get_array_action=lambda: FSEnvironment.load(
                path=file_path,
                adapter=adapter),
            tile=ImgLODPosition.from_string_rep(lod_str_rep)
        )
