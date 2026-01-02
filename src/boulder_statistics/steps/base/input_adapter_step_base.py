from dataclasses import dataclass

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment


@dataclass(frozen=True)
class InputAdapterStepBase[InputObjType, PathType: FSPathBase]():
    """Used to enforce name standardization"""
    input_adapter: FSAdapterBase[InputObjType, PathType]

    def get_input_object(self, path: PathType) -> InputObjType:
        return FSEnvironment.load(path, self.input_adapter)
