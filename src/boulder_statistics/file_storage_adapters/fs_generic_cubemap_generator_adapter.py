from dataclasses import dataclass, field
from os import walk
from os.path import join
from pathlib import Path
from typing import Dict

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.fs_generic_cubemap_generator import \
    FSGenericCubemapGenerator

FilePathLookupType = Dict[CubemapLodPosition, FSPathLocalDisk]


@dataclass(frozen=True)
class FSGenericCubemapGeneratorAdapter[T](
        FSAdapterBase[FSGenericCubemapGenerator, FSPathLocalDisk]):

    tiles_adapter: FSAdapterBase[T, FSPathLocalDisk]
    n_jobs: int = field(default=4)

    def read(
            self, path: FSPathLocalDisk) -> FSGenericCubemapGenerator:

        position_paths_lookup: FilePathLookupType = {}

        for dirpath, dirnames, filenames in walk(path.actual_path):
            for filename in filenames:
                full_path: Path = Path(join(dirpath, filename))

                if not CubemapLodPosition.is_correct_path_format(
                        path=full_path):
                    continue

                fs_full_path: FSPathLocalDisk = path.copy_from_folder(
                    new_sub_path=path.actual_path.relative_to(full_path)
                )

                cubemap_position: CubemapLodPosition = CubemapLodPosition.from_fs_path(
                    full_path)

                position_paths_lookup[cubemap_position] = fs_full_path

        return FSGenericCubemapGenerator(
            tiles=set(position_paths_lookup.keys()),
            generator_input=position_paths_lookup,
            adapter=self.tiles_adapter
        )

    def write(
            self, obj: FSGenericCubemapGenerator, path: FSPathLocalDisk) -> None:
        raise NotImplementedError
