from dataclasses import dataclass, field
from typing import Any

from joblib import delayed
from numpy import float64
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.steps.utils.Bennu_OBJ_to_LAS_cubemap_generator import \
    BennuOBJToLASCubemapGenerator
from boulder_statistics.steps.utils.PAN_to_LOD_cubemap_generator import \
    PANToLODCubemapGenerator

ArrayType = NDArray[float64]


@dataclass(frozen=True, kw_only=True)
class FSBennuObjToLODCubemapGeneratorAdapter(
        FSAdapterBase[BennuOBJToLASCubemapGenerator, FSPathLocalDisk]):

    tiles_adapter: FSAdapterBase[ArrayType, FSPathLocalDisk]
    n_jobs: int = field(default=4)
    standard_extension: str | None | bool = field(default=False)

    def read(
            self, path: FSPathLocalDisk) -> BennuOBJToLASCubemapGenerator:
        raise NotImplementedError

    def write(
            self, obj: CubemapGeneratorBase[Any, ArrayType], path: FSPathLocalDisk) -> None:

        ParallelPbar("Exporting tiles from cubemap generator", unit="tile")(n_jobs=self.n_jobs)(
            delayed(
                FSBennuObjToLODCubemapGeneratorAdapter.export_tile)(
                obj, path, tile, self.tiles_adapter)
            for tile in obj.tiles
        )

    @staticmethod
    def export_tile(
            obj: CubemapGeneratorBase[Any, ArrayType], path: FSPathLocalDisk,
            tile: CubemapLodPosition, tiles_adapter: FSAdapterBase[ArrayType, FSPathLocalDisk]) -> None:

        tile_export_object: ArrayType = obj.get_lod_tile(tile)
        tile_export_path: FSPathLocalDisk = tile.get_fs_path(
            root_path=path,
            markers=path.markers,
            tile_shape=tile_export_object.shape,
        )

        FSEnvironment.save(
            tile_export_object,
            tile_export_path,
            tiles_adapter)
