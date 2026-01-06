from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, List, Tuple

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.step_default_markers import StepDefaultMarkers
from boulder_statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class ManyToManyStepBase[ProcessJobInputObjectsType, ProcessJobOutputObjectsType](
    TaskStepBase,
    StepDefaultMarkers,
):
    """General many to many task base class backend"""

    pipeline_data_path: Path
    n_jobs: int = field(default_factory=lambda: 4, repr=False)
    loading_message: str = field(default="Running task...", repr=False)
    loading_unit: str = field(default="files", repr=False)
    folder_name_hash_length: int = field(default=8)

    def process_job(self,
                    input_object: ProcessJobInputObjectsType) -> List[FSPathLocalDisk]:

        output_object: ProcessJobOutputObjectsType = self.job_operation(
            input_object)

        export_files: List[FSObject] = self.process_job_output_to_fs_objects(
            output_object)

        return [export_file.fs_path for export_file in export_files]

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_with_markers: List[FSPathLocalDisk] = self.get_files_with_markers(
            env)

        input_objects: List[ProcessJobInputObjectsType] = self.input_objects_from_paths(
            files_with_markers)

        export_files_packed: List[List[FSPathLocalDisk]] = self.run_in_parallel(
            function=self.process_job,
            inputs=input_objects,
            n_jobs=self.n_jobs,
            message=self.loading_message,
            unit=self.loading_unit,
        )

        export_files: Tuple[FSPathLocalDisk, ...] = tuple(
            chain.from_iterable(export_files_packed))

        return FSEnvironment(export_files)

    # Helper methods for getting paths

    @property
    def task_path(self) -> Path:
        return self.pipeline_data_path / \
            Path(
                f"""Pipeline data for task {
                    self.task_name} with input hash {
                    self.task_hash[:self.folder_name_hash_length]}""")

    def get_output_file_path[I, O](
            self,
            input_object: I,
            output_object: O,
            get_object_relative_export_path: Callable[[I, O], Tuple[str, ...]]

    ) -> Path:

        return self.task_path / \
            Path(
                *
                get_object_relative_export_path(
                    input_object,
                    output_object))

    def get_FSPath_from_path[I, O](
            self, input_object: I, output_object: O,
            get_object_relative_export_path: Callable[[I, O], Tuple[str, ...]],
            output_markers: Tuple[FSMarkerBase, ...]) -> FSPathLocalDisk:

        return FSPathLocalDisk(
            root_path=self.pipeline_data_path.as_posix(),
            path=self.get_output_file_path(
                input_object,
                output_object,
                get_object_relative_export_path).parts,
            markers=output_markers
        )

    # Abstract methods for job processing

    @abstractmethod
    def input_objects_from_paths(
            self, input_paths: List[FSPathLocalDisk]) -> List[ProcessJobInputObjectsType]:
        ...

    @abstractmethod
    def job_operation(
            self, input_objects: ProcessJobInputObjectsType) -> ProcessJobOutputObjectsType:
        """
        This job must include the exporting of the files in ProcessJobOutputObjectsType

        Args:
            input_objects (ProcessJobInputObjectsType): A object which encapsulates the inputs for the job

        Returns:
            ProcessJobOutputObjectsType: A object which encapsulates the outputs for the job
        """
        ...

    @abstractmethod
    def process_job_output_to_fs_objects(
            self, output: ProcessJobOutputObjectsType) -> List[FSObject[Any, FSPathLocalDisk]]:
        ...
