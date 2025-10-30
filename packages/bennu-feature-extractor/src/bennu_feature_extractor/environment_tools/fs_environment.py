from typing import List, Set

import attr

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@attr.define()
class FSEnvironment():
    paths: Set[FSPathBase]

    def save[ObjType, PathType: FSPathBase](
            self, obj: ObjType, path: PathType, adapter: FSAdapterBase[ObjType, PathType]) -> None:
        self.paths.add(path)
        adapter.write(obj, path)

    def load[ObjType, PathType: FSPathBase](
            self, path: PathType, adapter: FSAdapterBase[ObjType, PathType]) -> ObjType:
        return adapter.read(path)

    @classmethod
    def empty(cls) -> 'FSEnvironment':
        return cls(paths=set())

    @staticmethod
    def merge(envs: List['FSEnvironment']) -> 'FSEnvironment':
        merged_paths: Set[FSPathBase] = {p for e in envs for p in e.paths}
        return FSEnvironment(paths=merged_paths)
