from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Tuple

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.step_default_adapters import StepDefaultAdapters
from boulder_statistics.step_default_markers import StepDefaultMarkers
from boulder_statistics.steps.base.input_adapter_step_base import \
    InputAdapterStepBase
from boulder_statistics.steps.base.many_to_many_step_base import \
    ManyToManyStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase
from boulder_statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class OneToOneStepBase[ProcessJobInputObjectType, ProcessJobOutputObjectType](
        ManyToManyStepBase[FSObject[ProcessJobInputObjectType, FSPathLocalDisk],
                           FSObject[ProcessJobOutputObjectType, FSPathLocalDisk]],
        InputAdapterStepBase[ProcessJobInputObjectType, FSPathLocalDisk],
        OutputAdapterStepBase[ProcessJobOutputObjectType, FSPathLocalDisk],
):
    """General one to one task base class backend"""

    def input_objects_from_paths(
            self, input_paths: List[FSPathLocalDisk]) -> List[FSObject[ProcessJobInputObjectType, FSPathLocalDisk]]:
        return [
            FSObject(
                fs_path=path,
                fs_adapter=self.input_adapter
            ) for path in input_paths
        ]

    def process_job_output_to_fs_objects(
            self, output: FSObject[ProcessJobOutputObjectType, FSPathLocalDisk]) -> List[FSObject[Any, FSPathLocalDisk]]:
        return [output]

    def job_operation(
            self, input_objects: FSObject[ProcessJobInputObjectType, FSPathLocalDisk]) -> FSObject[ProcessJobOutputObjectType, FSPathLocalDisk]:
        """
        This job must include the exporting of the files in ProcessJobOutputObjectsType

        Args:
            input_objects (ProcessJobInputObjectsType): A object which encapsulates the inputs for the job

        Returns:
            ProcessJobOutputObjectsType: A object which encapsulates the outputs for the job
        """

        output_object: ProcessJobOutputObjectType = self.object_operation(
            input_objects.object)

        return FSObject(
            fs_path=self.get_FSPath_from_path(
                input_objects.object,
                output_object,
                self.get_object_relative_export_path,
                output_markers=self.output_markers),
            fs_adapter=self.output_adapter,
        )

    @abstractmethod
    def get_object_relative_export_path(
            self, input_object: ProcessJobInputObjectType, output_object: ProcessJobOutputObjectType) -> Tuple[str, ...]:
        ...

    @abstractmethod
    def object_operation(self,
                         input_object: ProcessJobInputObjectType) -> ProcessJobOutputObjectType:
        ...
