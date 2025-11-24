from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import pandas as pd
from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.task_step_base import TaskStepBase

from bennu_feature_extractor_BoulderNet.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from bennu_feature_extractor_BoulderNet.file_storage_adapters.npz_detection_adapter import \
    FSNpzDetectionAdapter
from bennu_feature_extractor_BoulderNet.file_storage_adapters.pandas_pickle_adapter import \
    FSPandasPickleAdapter


@dataclass(frozen=True)
class DetectionMerge(TaskStepBase):
    marker_to_merge: FSMarkerString
    output_marker: FSMarkerString
    run_path: str
    result_output_path: str

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_merge: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: self.marker_to_merge in x.markers)

        detection_results: List[List[NpzFeatureDetection]] = [
            FSEnvironment.load(file, FSNpzDetectionAdapter()) for file in files_to_merge]

        merged_detections: pd.DataFrame = self.merge_detections(
            detection_results)

        result_path = FSPathLocalDisk(
            path=Path(self.result_output_path).parts,
            markers=frozenset({self.output_marker}),
            root_path=self.run_path,
        )

        FSEnvironment.save(
            merged_detections,
            result_path,
            FSPandasPickleAdapter())

        return FSEnvironment(frozenset([result_path]))

    @staticmethod
    def merge_detections(
            detections_list: List[List[NpzFeatureDetection]]) -> pd.DataFrame:

        return pd.DataFrame({
            "face": [detection.face for detections in detections_list for detection in detections],
            "relative_offset": [detection.relative_offset for detections in detections_list for detection in detections],
            "relative_scale": [detection.relative_scale for detections in detections_list for detection in detections],
            "box_xyxy": [detection.box_xyxy for detections in detections_list for detection in detections],
            "score": [detection.score for detections in detections_list for detection in detections],
            "class_id": [detection.class_id for detections in detections_list for detection in detections],
            "mask_uint8": [detection.mask_uint8 for detections in detections_list for detection in detections],
        })
