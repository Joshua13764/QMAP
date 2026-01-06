from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.iio_adapter import FSIIOAdapter
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.file_storage_adapters.shutil_copy_adapter import \
    FSShutilCopyAdapter
from boulder_statistics.lods.cubemap_generator_base import CubemapGeneratorBase
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.fs_copy_cubemap_generator import \
    FSCopyCubemapGenerator
from boulder_statistics.lods.fs_cubemap_generator import FSCubemapGenerator
from boulder_statistics.lods.lod_cubemap_utils import LODCubemapUtils
from boulder_statistics.steps.base.one_to_many_step_base import \
    OneToManyStepBase
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase
from boulder_statistics.steps.pds4_boulderNet_inference import \
    FSPathLocalDiskChunk
from boulder_statistics.steps.utils.pan_to_cubemap import PANToCubemap
from boulder_statistics.steps.utils.PAN_to_LOD_cubemap_generator import \
    PANToLODCubemapGenerator

ArrayType = NDArray[np.float64]


@dataclass(frozen=True)
class BetterPDS4BoulderNetInference(
        OneToOneStepBase[FSCopyCubemapGenerator, FSCopyCubemapGenerator]):

    cuda: bool = field(default=True)
    skip_if_exists: bool = field(default=True)
    append_input_extension_no_dot: str | None = field(
        default_factory=lambda: None)

    detection_export_custom_name_tag: str = field(default_factory=lambda: "")
    batch_size: int = 4096

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.task_name, self.skip_if_exists)

    def get_object_relative_export_path(
            self, input_object: FSCopyCubemapGenerator, output_object: CubemapGeneratorBase) -> Tuple[str, ...]:
        return ("detections",)

    def object_operation(
            self, input_object: FSCopyCubemapGenerator) -> FSCopyCubemapGenerator:

        self.copy_cubemap_structure_to_work_folder(input_object)
        overlay_output_files, detections_output_files = self.infer_flattened_cubemap_structure(
            input_object)

        return FSCopyCubemapGenerator(
            tiles=input_object.tiles,
            generator_input={tile: detections_output_file for tile, detections_output_file in zip(
                input_object.tiles, detections_output_files)},
            array_adapter=FSShutilCopyAdapter(overwrite=True))

    def infer_flattened_cubemap_structure(
            self, cubemap_structure: FSCopyCubemapGenerator) -> Tuple[List[FSPathLocalDisk], List[FSPathLocalDisk]]:

        flattened_paths: List[FSPathLocalDisk] = [self.get_flatted_cubemap_tile_path(
            tile) for tile in cubemap_structure.tiles]

        files_to_infer_with_extension: List[FSPathLocalDisk] = [
            f.copy_with_extension(self.append_input_extension_no_dot)
            if self.append_input_extension_no_dot is not None else f
            for f in flattened_paths
        ]

        inference_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=Path(f.root_path),
                # BoulderNetInference (bni)
                new_extension=f"{self.detection_export_custom_name_tag}.bni"
            )
            for f in files_to_infer_with_extension
        ]

        overlay_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=Path(f.root_path),
                new_extension=f"{
                    self.detection_export_custom_name_tag}_overlay.png"
            )
            for f in files_to_infer_with_extension
        ]

        detections_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new_name(
                new_root_path=Path(f.root_path),
                new_extension=f"{
                    self.detection_export_custom_name_tag}_detections.npz",
                markers=tuple()
            )
            for f in files_to_infer_with_extension
        ]

        exits_overlay: Dict[FSPathLocalDisk, bool] = FSEnvironment.quick_exists(
            overlay_output_files)
        exists_detections: Dict[FSPathLocalDisk, bool] = FSEnvironment.quick_exists(
            detections_output_files)

        def actual_files(files): return [
            file
            for file, overlay, detections in zip(files, overlay_output_files, detections_output_files)
            if not (exits_overlay[overlay] and exists_detections[detections] and self.skip_if_exists)
        ]

        if self.cuda:
            from boulder_statistics.steps.utils.docker_helpersCUDA import \
                DockerHelpers
        else:
            from boulder_statistics.steps.utils.docker_helpers import \
                DockerHelpers

        DockerHelpers.ensure_image_exists()

        in_folder_data: Dict[str, FSPathLocalDiskChunk] = self.sort_data_by_folders(
            actual_files(files_to_infer_with_extension), actual_files(inference_output_files))

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

        return overlay_output_files, detections_output_files

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

    def get_overlay_output_file_paths(
            self, files_to_infer: List[FSPathLocalDisk]) -> List[FSPathLocalDisk]:
        return [
            f.copy_as_new_name(
                new_root_path=Path(f.root_path),
                new_extension=f"{f.actual_path.stem}_overlay"
            )
            for f in files_to_infer
        ]

    def get_detections_output_file_paths(
            self, files_to_infer: List[FSPathLocalDisk]) -> List[FSPathLocalDisk]:
        return [
            f.copy_as_new_name(
                new_root_path=Path(f.root_path),
                new_extension=f"{f.actual_path.stem}_detections"
            )
            for f in files_to_infer
        ]

    def get_flatted_cubemap_tile_path(
            self, tile: CubemapLodPosition) -> FSPathLocalDisk:

        return self.get_FSPath_from_path(
            input_object=tile,
            output_object=None,
            get_object_relative_export_path=lambda i, o: (
                "temp", tile.string_rep),
            output_markers=tuple()
        )

    def copy_cubemap_structure_to_work_folder(
            self, cubemap_structure: FSCopyCubemapGenerator) -> None:

        for tile in cubemap_structure.tiles:
            FSEnvironment.save(
                obj=cubemap_structure.get_lod_tile(tile),
                path=self.get_flatted_cubemap_tile_path(tile),
                adapter=FSShutilCopyAdapter()
            )
