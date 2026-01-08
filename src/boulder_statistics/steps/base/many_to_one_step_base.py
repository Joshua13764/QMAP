from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_input import FSInput
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.inferences_cubemap_grader import \
    FSInferencesCubemapGrader
from boulder_statistics.steps.base.input_adapter_step_base import \
    InputAdapterStepBase
from boulder_statistics.steps.base.many_to_many_step_base import \
    ManyToManyStepBase
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase

ArrayType = NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class SelectedToOneStepBase[ProcessJobOutputObjectType](
    ManyToManyStepBase[FSEnvironment,
                       FSObject[ProcessJobOutputObjectType, FSPathLocalDisk]],
    OutputAdapterStepBase[ProcessJobOutputObjectType, FSPathLocalDisk]
):
    input_markers = None

    def input_objects_from_paths(
            self, input_paths: List[FSPathLocalDisk]) -> List[FSEnvironment]:
        return [FSEnvironment(paths=tuple(input_paths))]

    def process_job_output_to_fs_objects(
            self, output: FSObject[ProcessJobOutputObjectType, FSPathLocalDisk]) -> List[FSObject[Any, FSPathLocalDisk]]:
        return [output]

    def job_operation(
            self, input_objects: FSEnvironment) -> FSObject[ProcessJobOutputObjectType, FSPathLocalDisk]:
        """
        This job must include the exporting of the files in ProcessJobOutputObjectsType

        Args:
            input_objects (ProcessJobInputObjectsType): A object which encapsulates the inputs for the job

        Returns:
            ProcessJobOutputObjectsType: A object which encapsulates the outputs for the job
        """

        output_value: ProcessJobOutputObjectType = self.selected_objects_operation(
            input_objects)

        output_object: FSObject[ProcessJobOutputObjectType, FSPathLocalDisk] = FSObject(
            fs_path=self.get_FSPath_from_path(
                input_objects,
                output_value,
                self.get_object_relative_export_path,
                output_markers=self.output_markers),
            fs_adapter=self.output_adapter,
        )

        output_object.save_object(output_value)

        return output_object

    @abstractmethod
    def selected_objects_operation(self,
                                   env: FSEnvironment) -> ProcessJobOutputObjectType:
        ...
