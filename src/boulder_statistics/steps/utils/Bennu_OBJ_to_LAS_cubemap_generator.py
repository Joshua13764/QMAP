from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.steps.utils.plot_settings import PlotSettings
from boulder_statistics.steps.utils.projection_plotting import \
    ProjectionPlotting

ArrayType = NDArray[np.float64]

PlotSettings.load_default()


@dataclass
class BennuOBJToLASCubemapGenerator(
        CubemapGeneratorBase[PLOBJData, ArrayType]):
    tile_resolution: int
    colour_column_name: Callable[[str], str] = lambda face: f'{face}_ratio'

    def get_lod_tile(
            self, cubemaps_tile: CubemapLodPosition) -> ArrayType:

        return ProjectionPlotting.rasterize_tris(
            self.generator_input.verts, self.generator_input.tris, cubemaps_tile.face, cubemaps_tile.x_range, cubemaps_tile.y_range,
            (self.tile_resolution, self.tile_resolution), colour_column_name=self.colour_column_name)
