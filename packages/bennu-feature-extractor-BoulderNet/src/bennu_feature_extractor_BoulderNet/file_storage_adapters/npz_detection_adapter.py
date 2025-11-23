from typing import Any, List

import numpy as np
from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk

from bennu_feature_extractor_BoulderNet.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection


class FSNpzDetectionAdapter(
        FSAdapterBase[List[NpzFeatureDetection], FSPathLocalDisk]):
    """Uses the np module to load detection data"""

    def read(self, path: FSPathLocalDisk) -> List[NpzFeatureDetection]:
        data: Any = np.load(path.actual_path.as_posix())

        return [NpzFeatureDetection(*detection_data) for detection_data in zip(
            data["boxes_xyxy"],
            data["scores"],
            data["class_ids"],
            data["masks_uint8"],
        )]

    def write(self, obj: List[NpzFeatureDetection],
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
