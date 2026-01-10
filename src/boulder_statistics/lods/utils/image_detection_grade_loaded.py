from dataclasses import dataclass, field
from turtle import position
from typing import Any, List

from numpy import dtype
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.inference_detection_data import \
    InferenceDetectionData
from boulder_statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.utils.image_detection_grade import \
    ImageDetectionGrade


@dataclass(kw_only=True)
class ImageDetectionGradeLoaded():
    image_array: NDArray[Any]
    detection_data: InferenceDetectionData
    position: CubemapLodPosition

    @classmethod
    def from_grade(cls, grade: ImageDetectionGrade,
                   image_adapter: FSAdapterBase[NDArray[Any], FSPathLocalDisk],
                   detection_adapter: FSAdapterBase[List[InferenceDetectionData], FSPathLocalDisk]) -> "ImageDetectionGradeLoaded":

        return cls(
            image_array=FSEnvironment.load(
                grade.image_path, image_adapter),
            detection_data=FSEnvironment.load(
                grade.image_path, detection_adapter)[grade.detection_index],
            position=grade.cubemap_position
        )
