from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor.task_step_base import TaskStepBase
from bennu_feature_extractor_PDS.file_storage_adapters.pds4_adapter import (
    ArrayStructure, FSPDS4Adapter)
from joblib import delayed
from more_itertools import chunked
from numpy import dtype, ndarray
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor_BoulderNet.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from bennu_feature_extractor_BoulderNet.file_storage_adapters.npz_detection_adapter import \
    FSNpzDetectionAdapter
from bennu_feature_extractor_BoulderNet.utils import docker_helpers
from bennu_feature_extractor_BoulderNet.utils.docker_helpers import \
    DockerHelpers


@dataclass(frozen=True)
class DetectionMerge(TaskStepBase):
    marker_to_merge: FSMarkerString
    output_marker: FSMarkerString

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_merge: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: self.marker_to_merge in x.markers)

        detection_results: List[List[NpzFeatureDetection]] = [
            FSEnvironment.load(file, FSNpzDetectionAdapter()) for file in files_to_merge]
