from bennu_feature_extractor.task_step_base import TaskStepBase
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

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
class FSPathLocalDiskChunk():
    files_to_infer: List[FSPathLocalDisk] = field(default_factory=list)
    inference_output_files: List[FSPathLocalDisk] = field(default_factory=list)

    def get_sub_chunks(
            self, batch_size: int = 64) -> Iterator[Tuple[list[FSPathLocalDisk], list[FSPathLocalDisk]]]:

        return zip(chunked(self.files_to_infer, batch_size),
                   chunked(self.inference_output_files, batch_size))


@dataclass(frozen=True)
class PDS4BoulderNetInference(TaskStepBase):
    run_path: Path
    batch_size: int = 64

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

        in_folder_data: Dict[str, FSPathLocalDiskChunk] = self.sort_data_by_folders(
            files_to_infer, inference_output_files)

        for chunk_folder_path, chunk in in_folder_data.items():

            ParallelPbar(f"Inferring from images with batch size {self.batch_size} and {self.batch_size * len(chunk.inference_output_files)} images in folder {chunk_folder_path}",
                         unit=f"{self.batch_size} img batches")(n_jobs=1)(
                delayed(
                    DockerHelpers.analyse_image)(
                    image_paths,
                    inference_output_paths,
                    verbose=True)
                for image_paths, inference_output_paths in chunk.get_sub_chunks(batch_size=self.batch_size)
            )

        outputs: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=self.run_path,
                new_extension="_overlay.png"  # BoulderNetInference (bni)
            )
            for f in files_to_infer
        ] + [
            f.copy_as_new_name(
                new_root_path=self.run_path,
                new_extension="_detections.png"  # BoulderNetInference (bni)
            )
            for f in files_to_infer
        ]

        return FSEnvironment(frozenset(outputs))

    @staticmethod
    def sort_data_by_folders(files_to_infer: List[FSPathLocalDisk],
                             inference_output_files: List[FSPathLocalDisk]) -> Dict[str, FSPathLocalDiskChunk]:

        in_folders: Set[Path] = {
            file.actual_path.parent for file in files_to_infer}

        in_folders_data: Dict[str, FSPathLocalDiskChunk] = {
            folder.as_posix(): FSPathLocalDiskChunk() for folder in in_folders}

        for file_to_infer, inference_output_file in zip(
                files_to_infer, inference_output_files):

            in_folders_data[file_to_infer.actual_path.parent.as_posix(
            )].files_to_infer.append(file_to_infer)
            in_folders_data[file_to_infer.actual_path.parent.as_posix(
            )].inference_output_files.append(inference_output_file)

        return in_folders_data
