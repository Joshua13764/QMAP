from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import polars as pl
from numpy._typing._array_like import NDArray
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.img_lod_position import ImgLODPosition
from boulder_statistics.lods.img_lod_tile import LODImageTile
from boulder_statistics.steps.utils.projection_plotting import \
    ProjectionPlotting


@dataclass(frozen=True)
class LodFromProjectionRenderer():
    output_markers: Tuple[FSMarkerBase, ...]
    adapter: FSAdapterBase[NDArray[np.float64], FSPathLocalDisk]
    face: str
    tile: ImgLODPosition
    points: pl.LazyFrame
    tris: pl.LazyFrame
    face_lods_save_folder: FSPathLocalDisk
    resolution: int = field(default=512)
    verbose: bool = field(default=False)
    skip_if_exists: bool = field(default=True)
    colour_column_name: Callable[[str], str] = field(
        default=lambda face: f'{face}_ratio')

    @property
    def array_shape(self) -> Tuple[int, int]:
        return (self.resolution, self.resolution)

    def render_lod(self) -> LODImageTile[np.float64]:

        rendered_lod: NDArray[np.float64] = ProjectionPlotting.rasterize_tris(
            self.points, self.tris, self.face, self.tile.x_range, self.tile.y_range,
            self.array_shape)

        if self.verbose:
            print("Rasterized LOD tris")

        lod_tile: LODImageTile[np.float64] = LODImageTile[np.float64](
            tile=self.tile,
            array_storage_folder_location=self.face_lods_save_folder.copy_from_folder(
                Path("faces", f"face {self.face}"), self.output_markers
            ),
            array_storage_adapter=self.adapter,
            array_storage_markers=self.output_markers,
            array_memory=rendered_lod,
        )

        if self.verbose:
            print("Created tile")

        # To reduce memory usage save and unload
        lod_tile.unload_array_from_memory(save_if_in_memory=True)

        if self.verbose:
            print("Unloading from memory")

        return lod_tile
