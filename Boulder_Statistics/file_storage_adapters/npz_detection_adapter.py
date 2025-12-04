from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from Boulder_Statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection


@dataclass(frozen=True)
class FSNpzDetectionAdapter(
        FSAdapterBase[List[NpzFeatureDetection], FSPathLocalDisk]):
    """Uses the np module to load detection data"""

    def read(self, path: FSPathLocalDisk) -> List[NpzFeatureDetection]:
        data: Any = np.load(path.actual_path.as_posix())

        return [NpzFeatureDetection(
            FSNpzDetectionAdapter.find_face(
                path.actual_path), FSNpzDetectionAdapter.find_relative_offset(
                path.actual_path),
            FSNpzDetectionAdapter.find_relative_scale(path.actual_path), *detection_data) for detection_data in zip(
            data["boxes_xyxy"],
            data["scores"],
            data["class_ids"],
            data["masks_uint8"],
        )]

    def write(self, obj: List[NpzFeatureDetection],
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()

    @staticmethod
    def find_face(detection_path: Path) -> str:
        return detection_path.stem.split("_")[0]

    @staticmethod
    def find_relative_offset(detection_path: Path) -> NDArray[Any]:
        face, pixel_offset_x_str, pixel_offset_y_str, pixel_image_size, _, pixel_lod_size, *_ = detection_path.stem.split(
            "_")

        return np.array([
            float(pixel_offset_x_str) / float(pixel_lod_size),
            float(pixel_offset_y_str) / float(pixel_lod_size),
        ])

    @staticmethod
    def find_relative_scale(detection_path: Path) -> float:
        parent_name: str = detection_path.parent.name

        if parent_name.startswith("lod_"):
            return 2 ** -int(parent_name.replace("lod_", ""))
        else:
            raise ValueError(
                f"Cannot determine relative scale from parent folder name: {parent_name}"
            )
