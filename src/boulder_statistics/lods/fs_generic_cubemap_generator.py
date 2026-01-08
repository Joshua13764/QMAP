from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.based_cubemap_generator_base import \
    FSBasedCubemapGeneratorBase
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.lod_cubemap_utils import LODCubemapUtils
from boulder_statistics.steps.base.one_to_many_step_base import \
    OneToManyStepBase
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase
from boulder_statistics.steps.utils.pan_to_cubemap import PANToCubemap


@dataclass
class FSGenericCubemapGenerator[T](
        FSBasedCubemapGeneratorBase[T]):
    adapter: FSAdapterBase[T, FSPathLocalDisk]

    def get_lod_tile(
            self, cubemaps_tile: CubemapLodPosition) -> T:

        img: T = FSEnvironment.load(
            path=self.get_tile_path(cubemaps_tile),
            adapter=self.adapter,
        )

        return img
