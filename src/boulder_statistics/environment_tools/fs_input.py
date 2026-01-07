from dataclasses import dataclass
from typing import List

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSInput[ObjectType]():
    fs_marker: FSMarkerBase
    fs_adapter: FSAdapterBase[ObjectType, FSPathLocalDisk]

    def get_fs_path(self, env: FSEnvironment) -> FSPathLocalDisk:
        paths: List[FSPathLocalDisk] = env.get_paths_from_markers(
            FSPathLocalDisk, (self.fs_marker,))

        match len(paths):
            case 0: raise IndexError(f"No paths found in environment with type FSPathLocalDisk and a matching marker in {self.fs_marker}")
            case 1: return paths[0]
            case _:
                raise IndexError(
                    f"""Multiple paths ({paths}) found in environment with type FSPathLocalDisk and a matching marker in {
                        self.fs_marker}""")

    def get_fs_object(
            self, env: FSEnvironment) -> FSObject[ObjectType, FSPathLocalDisk]:
        return FSObject(
            fs_path=self.get_fs_path(env),
            fs_adapter=self.fs_adapter
        )

    def object(self, env: FSEnvironment) -> ObjectType:
        return self.get_fs_object(env).object

    def save_object(self, obj: ObjectType, env: FSEnvironment) -> None:
        return self.get_fs_object(env).save_object(obj)
