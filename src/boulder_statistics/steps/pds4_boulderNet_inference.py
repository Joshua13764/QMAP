from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from more_itertools import chunked
from tqdm import tqdm

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.task_step_base import TaskStepBase


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
    run_path: str
    batch_size: int = 64
    skip_converted: bool = field(default_factory=lambda: True)
    cuda: bool = field(default_factory=lambda: False)
    detection_input_markers: frozenset[FSMarkerString] = field(
        default_factory=lambda: frozenset([FSMarkerString("InferableImage")]))

    detection_output_markers: frozenset[FSMarkerString] = field(
        default_factory=lambda: frozenset([FSMarkerString("BoulderNet_Detections")]))

    detection_export_custom_name_tag: str = field(default_factory=lambda: "")

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_infer: List[FSPathLocalDisk] = env.get_paths_from_markers(
            FSPathLocalDisk, self.detection_input_markers)

        inference_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=Path(self.run_path),
                # BoulderNetInference (bni)
                new_extension=f"{self.detection_export_custom_name_tag}.bni"
            )
            for f in files_to_infer
        ]

        overlay_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=Path(self.run_path),
                new_extension=f"{
                    self.detection_export_custom_name_tag}_overlay.png"
            )
            for f in files_to_infer
        ]

        detections_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=Path(self.run_path),
                new_extension=f"{
                    self.detection_export_custom_name_tag}_detections.npz",
                markers=list(self.detection_output_markers)
            )
            for f in files_to_infer
        ]

        exits_overlay: Dict[FSPathLocalDisk, bool] = FSEnvironment.quick_exists(
            overlay_output_files)
        exists_detections: Dict[FSPathLocalDisk, bool] = FSEnvironment.quick_exists(
            detections_output_files)

        def actual_files(files): return [
            file
            for file, overlay, detections in zip(files, overlay_output_files, detections_output_files)
            if not (exits_overlay[overlay] and exists_detections[detections] and self.skip_converted)
        ]

        if self.cuda:
            from boulder_statistics.steps.utils.docker_helpersCUDA import \
                DockerHelpers
        else:
            from boulder_statistics.steps.utils.docker_helpers import \
                DockerHelpers

        DockerHelpers.ensure_image_exists()

        in_folder_data: Dict[str, FSPathLocalDiskChunk] = self.sort_data_by_folders(
            actual_files(files_to_infer), actual_files(inference_output_files))

        for chunk_folder_path, chunk in in_folder_data.items():

            sub_chunks = list(chunk.get_sub_chunks(batch_size=self.batch_size))

            for image_paths, inference_output_paths in tqdm(
                sub_chunks,
                desc=(
                    f"Inferring from images with batch size {
                        self.batch_size} and "
                    f"{len(chunk.inference_output_files)} images in folder {chunk_folder_path}"
                ),
                unit=f"{self.batch_size} img batches",
                total=len(sub_chunks),
            ):
                DockerHelpers.analyse_image(
                    image_paths,
                    inference_output_paths,
                    verbose=True,
                    detection_export_custom_name_tag=self.detection_export_custom_name_tag
                )

        return FSEnvironment(
            frozenset(overlay_output_files + detections_output_files))

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
