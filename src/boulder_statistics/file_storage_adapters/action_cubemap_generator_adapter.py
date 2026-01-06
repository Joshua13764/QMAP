from dataclasses import dataclass, field
from os import walk
from os.path import join
from pathlib import Path
from typing import Dict

from numpy import float64
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.fs_cubemap_generator import FSCubemapGenerator

FilePathLookupType = Dict[CubemapLodPosition, FSPathLocalDisk]
ArrayType = NDArray[float64]


@dataclass(frozen=True)
class FSActionCubemapGeneratorAdapter(
        FSAdapterBase[FSCubemapGenerator, FSPathLocalDisk]):

    tiles_adapter: FSAdapterBase[ArrayType, FSPathLocalDisk]
    n_jobs: int = field(default=4)

    def read(
            self, path: FSPathLocalDisk) -> FSCubemapGenerator:

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

        return FSCubemapGenerator(
            tiles=set(position_paths_lookup.keys()),
            generator_input=position_paths_lookup,
            array_adapter=self.tiles_adapter
        )

    def write(
            self, obj: FSCubemapGenerator, path: FSPathLocalDisk) -> None:
        raise NotImplementedError
