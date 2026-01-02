from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.step_default_adapters import StepDefaultAdapters
from boulder_statistics.step_default_markers import StepDefaultMarkers
from boulder_statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class OneToOneStepBase[InputObjType, OutputObjType](
        TaskStepBase, StepDefaultMarkers, StepDefaultAdapters[InputObjType, OutputObjType, FSPathLocalDisk]):
    pipeline_data_path: Path
    n_jobs: int = field(default_factory=lambda: 4, repr=False)
    loading_message: str = field(default="Running task...", repr=False)
    loading_unit: str = field(default="files", repr=False)

    @property
    def task_path(self) -> Path:
        return self.pipeline_data_path / \
            Path(
                f"""Pipeline data for task {
                    self.task_name} with input hash {
                    self.task_hash}""")

    @property
    def core_task_operation(
            self) -> Callable[[FSPathLocalDisk], FSPathLocalDisk]:

        def operation(input_path: FSPathLocalDisk) -> FSPathLocalDisk:
            input_object: InputObjType = self.get_input_object(input_path)
            output_object: OutputObjType = self.object_operation(input_object)

            output_path: FSPathLocalDisk = self.get_FSPath_from_path(
                input_object, output_object)

            return output_path

        return operation

    def get_output_file_path(self, input_object: InputObjType,
                             output_object: OutputObjType) -> Path:
        return self.task_path / \
            Path(
                *
                self.get_object_relative_export_path(
                    input_object,
                    output_object))

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_with_markers: List[FSPathLocalDisk] = self.get_files_with_markers(
            env)

        export_files: List[FSPathLocalDisk] = self.run_in_parallel(
            function=self.core_task_operation,
            inputs=files_with_markers,
            n_jobs=self.n_jobs,
            message=self.loading_message,
            unit=self.loading_unit,
        )

        return FSEnvironment(tuple(export_files))

    def get_FSPath_from_path(
            self, input_object: InputObjType, output_object: OutputObjType) -> FSPathLocalDisk:

        return FSPathLocalDisk(
            root_path=self.pipeline_data_path.as_posix(),
            path=self.get_output_file_path(input_object, output_object).parts,
            markers=self.output_markers
        )

    @abstractmethod
    def object_operation(input_object: InputObjType) -> OutputObjType:
        ...

    @abstractmethod
    def get_object_relative_export_path(
            self, input_object: InputObjType, output_object: OutputObjType) -> Tuple[str, ...]:
        ...
