from dataclasses import dataclass

from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.steps.base.input_adapter_step_base import \
    InputAdapterStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase


@dataclass(frozen=True)
class StepDefaultAdapters[InputObjType, OutputObjType, PathType: FSPathBase](
    InputAdapterStepBase[InputObjType, PathType],
    OutputAdapterStepBase[OutputObjType, PathType]
):
    """Used to enforce name standardization"""
