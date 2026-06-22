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
from boulder_statistics.steps.utils.PAN_to_LOD_cubemap_generator import \
    PANToLODCubemapGenerator

ArrayType = NDArray[np.float64]


@dataclass(frozen=True)
class BetterPANToLOD(
        OneToOneStepBase[ArrayType, PANToLODCubemapGenerator]):

    lod_depth: int = field(default_factory=lambda: 4)
    lod_res: int = field(default_factory=lambda: 512)
    super_sample_factor: int = field(default_factory=lambda: 4)
    skip_if_exists: bool = field(default_factory=lambda: True)

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.task_name,)

    def get_object_relative_export_path(
            self, input_object: ArrayType, output_object: PANToLODCubemapGenerator) -> Tuple[str, ...]:
        return (
            f"""cubemap_lod_depth ({
                self.lod_depth}) res ({
                self.lod_res}) super_sample_factor ({
                self.super_sample_factor})""",
        )

    def object_operation(
            self, input_object: ArrayType) -> PANToLODCubemapGenerator:
        return PANToLODCubemapGenerator(
            tiles=LODCubemapUtils.get_all_cubemap_tiles_for_depths(
                max_depth=self.lod_depth),
            tile_resolution=self.lod_res,
            tile_super_sample_factor=self.super_sample_factor,
            generator_input=input_object
        )
