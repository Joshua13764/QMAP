from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List

from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.steps.base.input_adapter_step_base import \
    InputAdapterStepBase
from boulder_statistics.steps.base.many_to_many_step_base import \
    ManyToManyStepBase


@dataclass(frozen=True)
class OneToManyStepBase[ProcessJobInputObjectType, ProcessJobOutputObjectsType](
        ManyToManyStepBase[FSObject[ProcessJobInputObjectType, FSPathLocalDisk],
                           ProcessJobOutputObjectsType],
        InputAdapterStepBase[ProcessJobInputObjectType, FSPathLocalDisk]
):
    """General one to many task base class backend"""

    def input_objects_from_paths(
            self, input_paths: List[FSPathLocalDisk]) -> List[FSObject[ProcessJobInputObjectType, FSPathLocalDisk]]:
        return [
            FSObject(
                fs_path=path,
                fs_adapter=self.input_adapter
            ) for path in input_paths
        ]

    @abstractmethod
    def job_operation(
            self, input_objects: FSObject[ProcessJobInputObjectType, FSPathLocalDisk]) -> ProcessJobOutputObjectsType:
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
