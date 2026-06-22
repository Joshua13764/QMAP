from dataclasses import dataclass

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment


@dataclass(frozen=True)
class FSObject[ObjectType, PathType: FSPathBase]():
    fs_path: PathType
    fs_adapter: FSAdapterBase[ObjectType, PathType]

    @property
    def object(self) -> ObjectType:
        return FSEnvironment.load(self.fs_path, self.fs_adapter)

    def save_object(self, obj: ObjectType) -> None:
        FSEnvironment.save(obj, self.fs_path, self.fs_adapter)
