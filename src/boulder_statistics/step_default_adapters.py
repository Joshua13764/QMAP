from dataclasses import dataclass, field
from typing import Any, List, Tuple

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class StepDefaultAdapters[InputObjType,
                          OutputObjType, PathType: FSPathBase]():
    input_adapter: FSAdapterBase[InputObjType, PathType]
    output_adapter: FSAdapterBase[OutputObjType, PathType]

    def get_input_object(self, path: PathType) -> InputObjType:
        return FSEnvironment.load(path, self.input_adapter)

    def save_output_object(
            self, output_object: OutputObjType, path: PathType) -> None:
        FSEnvironment.save(output_object, path, self.output_adapter)
