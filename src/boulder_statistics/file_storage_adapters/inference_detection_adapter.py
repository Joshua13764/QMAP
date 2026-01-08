from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.inference_detection_data import \
    InferenceDetectionData
from boulder_statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection


@dataclass(frozen=True)
class FSInferenceDetectionAdapter(
        FSAdapterBase[List[InferenceDetectionData], FSPathLocalDisk]):
    """Uses the np module to load detection data"""
    standard_extension = "npz"

    def read(self, path: FSPathLocalDisk) -> List[InferenceDetectionData]:
        data: Any = np.load(path.actual_path.as_posix())

        return [
            InferenceDetectionData(
                box_xyxy=bounding_box,
                score=score,
                class_id=class_id,
                mask_uint8=mask_unit8
            )
            for bounding_box, score, class_id, mask_unit8 in zip(
                data["boxes_xyxy"],
                data["scores"],
                data["class_ids"],
                data["masks_uint8"],
            )]

    def write(self, obj: List[InferenceDetectionData],
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
