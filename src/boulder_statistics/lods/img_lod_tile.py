from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.img_lod_position import ImgLODPosition

T = TypeVar("T", bound=np.floating)


@dataclass()
class LODImageTile(Generic[T]):
    tile: ImgLODPosition

    array_storage_folder_location: FSPathLocalDisk
    array_storage_adapter: FSAdapterBase[NDArray[T], FSPathLocalDisk]
    array_storage_markers: Tuple[FSMarkerBase, ...]
    array_shape: Tuple[int, ...]
    array_memory: NDArray[T] | None

    cache_array_loads: bool = field(default=True)
    auto_update_local_disk_upon_array_set: bool = field(default=True)

    @property
    def array(self) -> NDArray[T]:

        if self.array_memory is not None:
            return self.array_memory.copy()

        elif self.array_exist_in_local_disk == True and self.cache_array_loads == True:
            self.array = self.get_array_from_local_disk()
            return self.array

        elif self.array_exist_in_local_disk == True and self.cache_array_loads == False:
            return self.get_array_from_local_disk()

        elif self.array_exist_in_local_disk == False:
            raise NotADirectoryError(
                "Cannot find an array in either memory or local disk for the LODImageTile")

        else:
            raise NotImplementedError("Unhandled condition")

    @array.setter
    def array(self, value: NDArray[T]) -> None:

        if self.array_shape != value.shape:
            raise Exception("Array shape mismatch for setter")

        else:
            self.array_memory = value

            if self.auto_update_local_disk_upon_array_set:
                self.save_array_to_local_disk()

    @property
    def array_exist_in_local_disk(self) -> bool:
        return self.local_disk_save_path.exists

    @property
    def local_disk_save_path(self) -> FSPathLocalDisk:

        lod_str_rep: str = self.tile.string_rep
        lod_number: int = self.tile.lod_number

        rel_path: Path = Path(
            f"lod {lod_number}", f"lod tile {lod_str_rep} with shape {
                self.array_shape}"
        )

        local_disk_save_path: FSPathLocalDisk = self.array_storage_folder_location.copy_from_folder(
            rel_path, markers=self.array_storage_markers)

        return local_disk_save_path

    def unload_array_from_memory(self) -> None:
        self.array_memory = None

    def get_array_from_local_disk(self) -> NDArray[T]:
        return FSEnvironment.load(
            path=self.local_disk_save_path,
            adapter=self.array_storage_adapter
        )

    def save_array_to_local_disk(self, skip_if_unloaded_array=True) -> None:
        if self.array_memory is None and skip_if_unloaded_array:
            return

        FSEnvironment.save(
            obj=self.array,
            path=self.local_disk_save_path,
            adapter=self.array_storage_adapter
        )

    @classmethod
    def from_get_array_action(cls, tile: ImgLODPosition, array_storage_folder_location: FSPathLocalDisk, array_storage_adapter:
                              FSAdapterBase[NDArray[T], FSPathLocalDisk], array_storage_markers: Tuple[FSMarkerBase, ...],
                              get_array_action: Callable[[], NDArray[T]], array_shape: Tuple[int, ...], skip_if_exists=True) -> "LODImageTile[T]":

        shell: "LODImageTile[T]" = cls(
            tile=tile,
            array_storage_folder_location=array_storage_folder_location,
            array_storage_adapter=array_storage_adapter,
            array_storage_markers=array_storage_markers,
            array_shape=array_shape,
            array_memory=None)

        print(shell.array_exist_in_local_disk and skip_if_exists)

        shell.array = shell.get_array_from_local_disk(
        ) if shell.array_exist_in_local_disk and skip_if_exists else get_array_action()

        return shell
