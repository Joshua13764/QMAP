from dataclasses import dataclass

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment


@dataclass(frozen=True)
class OutputAdapterStepBase[OutputObjType, PathType: FSPathBase]():
    """Used to enforce name standardization"""
    output_adapter: FSAdapterBase[OutputObjType, PathType]

    def save_output_object(
            self, output_object: OutputObjType, path: PathType) -> None:
        FSEnvironment.save(output_object, path, self.output_adapter)
