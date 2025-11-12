from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor_PDS.file_storage_adapters.pds4_adapter import (
    ArrayStructure, FSPDS4Adapter)
from joblib import delayed
from more_itertools import chunked
from numpy import dtype, ndarray
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor_BoulderNet.utils import docker_helpers
from bennu_feature_extractor_BoulderNet.utils.docker_helpers import \
    DockerHelpers


@dataclass()
class PDS4BoulderNetInference(StepBase):
    run_path: Path
    batch_size: int = 5

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_infer: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: FSMarkerString("InferableImage") in x.markers)

        inference_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new(
                new_root_path=self.run_path,
                new_extension=".bni"  # BoulderNetInference (bni)
            )
            for f in files_to_infer
        ]

        DockerHelpers.ensure_image_exists()

        ParallelPbar(f"Inferring from images", unit="img batches")(n_jobs=1)(
            delayed(
                DockerHelpers.analyse_image)(
                image_paths,
                inference_output_paths,
                verbose=True)
            for image_paths, inference_output_paths in zip(
                chunked(files_to_infer, self.batch_size), chunked(inference_output_files, self.batch_size))
        )
