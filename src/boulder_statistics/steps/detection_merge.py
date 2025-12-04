from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
from joblib import delayed
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from boulder_statistics.file_storage_adapters.npz_detection_adapter import \
    FSNpzDetectionAdapter
from boulder_statistics.file_storage_adapters.pandas_pickle_adapter import \
    FSPandasPickleAdapter
from boulder_statistics.task_step_base import TaskStepBase

HEADERS: List[str] = ["face", "relative_offset", "relative_scale",
                      "box_xyxy", "score", "class_id", "fixed_weight_area"]


@dataclass(frozen=True)
class DetectionMerge(TaskStepBase):
    marker_to_merge: FSMarkerString
    output_marker: FSMarkerString
    run_path: str
    result_output_path: str

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_merge: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: self.marker_to_merge in x.markers)

        processed_Bouldernet_inferences = ParallelPbar(
            f"Processing BoulderNet {len(files_to_merge)} inferences", unit="inferences")(n_jobs=-1)(
            delayed(DetectionMerge.process_Bouldernet_inference)(path)
            for path in files_to_merge
        )

        merged_detections: pd.DataFrame = self.merge_inferences(
            processed_Bouldernet_inferences)

        result_path = FSPathLocalDisk(
            path=Path(self.result_output_path).parts,
            markers=frozenset([self.output_marker]),
            root_path=self.run_path,
        )

        self.logger.info(
            f"""Saving {
                merged_detections.shape[0]} merged detections to {
                result_path.actual_path.as_posix()}""")

        FSEnvironment.save(
            merged_detections,
            result_path,
            FSPandasPickleAdapter())

        return FSEnvironment(frozenset([result_path]))

    @staticmethod
    def process_Bouldernet_inference(
            Bouldernet_inference_path: FSPathLocalDisk) -> Dict[str, Any]:

        inference_results: List[NpzFeatureDetection] = FSEnvironment.load(
            Bouldernet_inference_path, FSNpzDetectionAdapter())

        inference_results_packaged: Dict[str, Any] = {
            "face": [detection.face for detection in inference_results],
            "relative_offset": [detection.relative_offset for detection in inference_results],
            "relative_scale": [detection.relative_scale for detection in inference_results],
            "box_xyxy": [detection.box_xyxy for detection in inference_results],
            "score": [detection.score for detection in inference_results],
            "class_id": [detection.class_id for detection in inference_results],
            # "mask_uint8" : [],
            "fixed_weight_area": [detection.get_area_fixed_weight(
                detection.relative_scale) for detection in inference_results],
        }

        return inference_results_packaged

    def merge_inferences(self,
                         inference_results: List[Dict[str, Any]]) -> pd.DataFrame:

        self.logger.info(f"Merging detections from all inferences...")

        flatten_results: Callable[[str], List[Any]] = lambda header_name: [
            i for inference_result in inference_results for i in inference_result[header_name]]

        return pd.DataFrame({header: flatten_results(header)
                            for header in HEADERS})
