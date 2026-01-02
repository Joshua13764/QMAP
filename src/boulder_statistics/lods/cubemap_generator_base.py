from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.lod_cubemap_utils import LODCubemapUtils
from boulder_statistics.steps.base.one_to_many_step_base import \
    OneToManyStepBase
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase
from boulder_statistics.steps.utils.pan_to_cubemap import PANToCubemap

ArrayType = NDArray[np.float64]


@dataclass(frozen=True)
class CubemapGeneratorBase[GeneratorInputType, TileDataType](ABC):
    tiles: Set[CubemapLodPosition]
    tile_resolution: int
    tile_super_sample_factor: int

    generator_input: GeneratorInputType

    @abstractmethod
    def render_lod_tile(
            self, cubemaps_tile: CubemapLodPosition) -> TileDataType:
        ...
