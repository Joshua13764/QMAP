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
    detection_index: int
    position: CubemapLodPosition

    @classmethod
    def all_from_detection(cls,
                           image_path: FSPathLocalDisk,
                           detections_path: FSPathLocalDisk,
                           image_adapter: FSAdapterBase[NDArray[Any], FSPathLocalDisk],
                           detection_adapter: FSAdapterBase[List[InferenceDetectionData], FSPathLocalDisk]) -> List["ImageDetectionGradeLoaded"]:

        img_array: NDArray[Any] = FSEnvironment.load(
            image_path, image_adapter)

        return [
            cls(
                image_array=img_array,
                detection_data=detection_data,
                detection_index=detection_index,
                position=CubemapLodPosition.from_fs_path(
                    image_path.actual_path),
            )
            for detection_index, detection_data in enumerate(FSEnvironment.load(
                detections_path, detection_adapter))
        ]
