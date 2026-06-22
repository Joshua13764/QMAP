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
from boulder_statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from boulder_statistics.file_storage_adapters.npz_detection_adapter import \
    FSNpzDetectionAdapter
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.lod_cubemap_utils import LODCubemapUtils
from boulder_statistics.steps.base.one_to_many_step_base import \
    OneToManyStepBase
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase
from boulder_statistics.steps.utils.pan_to_cubemap import PANToCubemap

ArrayType = NDArray[np.float64]

# @dataclass
# class NpzFeatureDetection():
#     face: str
#     relative_offset: NDArray[Any]
#     relative_scale: float
#     box_xyxy: NDArray[Any]
#     score: float
#     class_id: int
#     mask_uint8: NDArray[Any]

#     def get_area_fixed_weight(self, per_pixel_weight: float = 1) -> float:
#         return np.sum(self.mask_uint8) * per_pixel_weight

#     def get_area_variable_weight(self, weight_map: NDArray[Any]) -> float:
#         return np.sum(self.mask_uint8 * weight_map)

DetectionFeatureSet = List[NpzFeatureDetection]


@dataclass
class FSInferencesCubemapGrader(
        CubemapGeneratorBase[Dict[CubemapLodPosition, FSPathLocalDisk], DetectionFeatureSet]):

    tile_inference_adapter: FSAdapterBase[DetectionFeatureSet, FSPathLocalDisk] = field(
        default=FSNpzDetectionAdapter())

    def get_lod_tile(
            self, cubemaps_tile: CubemapLodPosition) -> DetectionFeatureSet:
        """Collects the grades associated with a given LOD tile

        Args:
            cubemaps_tile (CubemapLodPosition): The tile who's grades are being enquired

        Returns:
            DetectionFeatureSet: _description_
        """

        return FSEnvironment.load(
            path=self.generator_input[cubemaps_tile],
            adapter=self.tile_inference_adapter
        )
