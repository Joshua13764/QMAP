from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.utils.image_detection_data import \
    ImageDetectionData


@dataclass(frozen=True, kw_only=True)
class ImageDetectionGrade():
    # Path of where the src image originally taken
    image_path: FSPathLocalDisk

    # Index of the path of the detection inferences (detections_path) as well
    # as the index of the detection in that inference set (detection_index)
    detections_path: FSPathLocalDisk
    detection_index: int

    # # Depending on the folder will dictate the grade of the detection
    # graded_path: FSPathLocalDisk = field(default=)

    @property
    def detection_data(self) -> ImageDetectionData:
        raise NotImplementedError

    @property
    def image_data(self) -> NDArray[Any]:
        raise NotImplementedError

    @property
    def lod_position(self) -> CubemapLodPosition:
        raise NotImplementedError

    @property
    def detection_grade(self) -> float:
        raise NotImplementedError
