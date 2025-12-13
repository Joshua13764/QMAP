from dataclasses import dataclass, field
from email.policy import default
from pathlib import Path
from typing import Generic, TypeVar

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

    array_storage_folder_location: FSPathLocalDisk
    array_storage_adapter: FSAdapterBase[NDArray[T], FSPathLocalDisk]
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

    def unload_array_from_memory(self) -> None:
        self.array_memory = None

    @property
    def local_disk_save_path(self) -> FSPathLocalDisk:

        lod_str_rep: str = self.tile.string_rep
        lod_number: int = self.tile.lod_number

        rel_path: Path = Path(
            f"lod {lod_number}", f"lod tile {lod_str_rep} with shape {
                self.array.shape}"
        )

        local_disk_save_path: FSPathLocalDisk = self.array_storage_folder_location.copy_from_folder(
            rel_path)

        return local_disk_save_path

    @property
    def array_from_local_disk(self) -> NDArray[T]:
        return FSEnvironment.load(
            path=self.local_disk_save_path,
            adapter=self.array_storage_adapter
        )

    def save_array_to_local_disk(self) -> None:
        FSEnvironment.save(
            obj=self.array,
            path=self.local_disk_save_path,
            adapter=self.array_storage_adapter
        )

    # @classmethod
    # def from_file(cls, file_path: FSPathLocalDisk,
    #               adapter: FSAdapterBase) -> "LODImageTile[T]":
    #     lod_str_rep: str = file_path.actual_path.stem.split(" ")[2]

    #     return cls(
    #         get_array_action=lambda:),
    #     tile = ImgLODPosition.from_string_rep(lod_str_rep)
    #     )
