from dataclasses import dataclass, field
from email.policy import default
from pathlib import Path
from typing import Generic, Tuple, TypeVar

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
    array_memory: NDArray[T] | None = field(default=None)

    @property
    def array(self) -> NDArray[T]:
        """Used to lazily generate the array and then cache the result

        Returns:
            NDArray[T]: _description_
        """
        if self.array_memory is None:
            self.array_memory = self.array_from_local_disk

        return self.array_memory

    def set_array_from_memory(self, array: NDArray[T]) -> None:
        self.array_memory = array.copy()
        self.save_array_to_local_disk()

    def unload_array_from_memory(self, save_if_in_memory=True) -> None:
        if save_if_in_memory and self.array_memory is not None:
            self.save_array_to_local_disk()

        self.array_memory = None

    @property
    def local_disk_save_path(self) -> FSPathLocalDisk:

        lod_str_rep: str = self.tile.string_rep
        lod_number: int = self.tile.lod_number

        assert self.array is not None

        rel_path: Path = Path(
            f"lod {lod_number}", f"lod tile {lod_str_rep} with shape {
                self.array.shape}"
        )

        local_disk_save_path: FSPathLocalDisk = self.array_storage_folder_location.copy_from_folder(
            rel_path, markers=self.array_storage_markers)

        return local_disk_save_path

    @property
    def array_from_local_disk(self) -> NDArray[T]:
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
