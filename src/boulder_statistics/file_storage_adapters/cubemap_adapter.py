from dataclasses import dataclass
from os import walk
from os.path import join
from pathlib import Path
from typing import Any, Dict

from joblib import delayed
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.fs_cubemap_generator import FSCubemapGenerator


@dataclass(frozen=True)
class FSCubemapAdapter(
        FSAdapterBase[CubemapGeneratorBase[Any, NDArray[Any]], FSPathLocalDisk]):

    tiles_adapter: FSAdapterBase[NDArray[Any], FSPathLocalDisk]
    n_jobs: int = 4

    def read(
            self, path: FSPathLocalDisk) -> CubemapGeneratorBase[Any, NDArray[Any]]:

        position_paths_lookup: Dict[CubemapLodPosition, FSPathLocalDisk] = {}

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
            array_read_adapter=self.tiles_adapter
        )

    def write(
            self, obj: CubemapGeneratorBase[Any, NDArray[Any]], path: FSPathLocalDisk) -> None:

        ParallelPbar("Exporting tiles from cubemap generator", unit="tile")(n_jobs=self.n_jobs)(
            delayed(FSCubemapAdapter.export_tile)(tile)
            for tile in obj.tiles
        )

    @staticmethod
    def export_tile(
            obj: CubemapGeneratorBase[Any, NDArray[Any]], path: FSPathLocalDisk,
            tile: CubemapLodPosition, tiles_adapter: FSAdapterBase[NDArray[Any], FSPathLocalDisk]) -> None:

        tile_export_object: NDArray[Any] = obj.get_lod_tile(tile)
        tile_export_path: FSPathLocalDisk = tile.get_fs_path(
            root_path=path,
            markers=path.markers,
            tile_shape=tile_export_object.shape,
        )

        FSEnvironment.save(
            tile_export_object,
            tile_export_path,
            tiles_adapter)
