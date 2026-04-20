from dataclasses import dataclass, field
from os import walk
from os.path import join
from pathlib import Path
from typing import Any, ClassVar, Dict

from joblib import delayed
from numpy import float64
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.shutil_copy_adapter import \
    FSShutilCopyAdapter
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.fs_copy_cubemap_generator import \
    FSCopyCubemapGenerator
from boulder_statistics.steps.utils.PAN_to_LOD_cubemap_generator import \
    PANToLODCubemapGenerator

FilePathLookupType = Dict[CubemapLodPosition, FSPathLocalDisk]
ArrayType = NDArray[float64]


@dataclass(frozen=True)
class FSCopyCubemapGeneratorAdapter(
        FSAdapterBase[FSCopyCubemapGenerator, FSPathLocalDisk]):

    tiles_adapter: FSAdapterBase[FSPathLocalDisk, FSPathLocalDisk] = field(
        default_factory=lambda: FSShutilCopyAdapter())
    standard_extension: str | None | bool = field(default=False)
    n_jobs: int = field(default=4)

    def read(
            self, path: FSPathLocalDisk) -> FSCopyCubemapGenerator:

        position_paths_lookup: FilePathLookupType = {}

        for dirpath, dirnames, filenames in walk(path.actual_path):
            for filename in filenames:
                full_path: Path = Path(join(dirpath, filename))

                if not CubemapLodPosition.is_correct_path_format(
                        path=full_path):
                    continue

                fs_full_path: FSPathLocalDisk = path.copy_from_folder(
                    new_sub_path=full_path.relative_to(path.actual_path)
                )

                cubemap_position: CubemapLodPosition = CubemapLodPosition.from_fs_path(
                    full_path)

                position_paths_lookup[cubemap_position] = fs_full_path

        return FSCopyCubemapGenerator(
            tiles=set(position_paths_lookup.keys()),
            generator_input=position_paths_lookup,
        )


    def write(
            self, obj: FSCopyCubemapGenerator, path: FSPathLocalDisk) -> None:

        ParallelPbar("Coping tiles from cubemap generator", unit="tile")(n_jobs=self.n_jobs)(
            delayed(
                FSCopyCubemapGeneratorAdapter.export_tile)(
                obj, path, tile, self.tiles_adapter)
            for tile in obj.tiles
        )

    @staticmethod
    def export_tile(
            obj: FSCopyCubemapGenerator, path: FSPathLocalDisk,
            tile: CubemapLodPosition, tiles_adapter: FSAdapterBase[FSPathLocalDisk, FSPathLocalDisk]) -> None:

        tile_export_src: FSPathLocalDisk = obj.get_lod_tile(tile)
        tile_export_path: FSPathLocalDisk = tile.get_fs_path(
            root_path=path,
            markers=path.markers,
            tile_shape=(0, 0),
        )

        FSEnvironment.save(
            tile_export_src,
            tile_export_path,
            tiles_adapter)
