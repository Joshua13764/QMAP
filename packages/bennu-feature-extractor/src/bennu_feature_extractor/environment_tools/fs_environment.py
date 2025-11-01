from pathlib import Path
from typing import Callable, List, Set

import attr
from jinja2 import Environment

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@attr.define(frozen=True, slots=True)
class FSEnvironment():
    paths: frozenset[FSPathBase]

    def get_paths[T: FSPathBase](self, cls: type[T], condition: Callable[[
                                 T], bool] = lambda x: True) -> List[T]:
        return [f for f in self.paths if isinstance(f, cls) and condition(f)]

    @classmethod
    def empty(cls) -> 'FSEnvironment':
        return cls(paths=frozenset())

    @staticmethod
    def save[ObjType, PathType: FSPathBase](
            obj: ObjType, path: PathType, adapter: FSAdapterBase[ObjType, PathType]) -> None:
        adapter.write(obj, path)

    @staticmethod
    def load[ObjType, PathType: FSPathBase](
            path: PathType, adapter: FSAdapterBase[ObjType, PathType]) -> ObjType:
        return adapter.read(path)

    @staticmethod
    def merge(envs: List['FSEnvironment']) -> 'FSEnvironment':
        merged_paths: frozenset[FSPathBase] = frozenset({
            p for e in envs for p in e.paths})
        return FSEnvironment(paths=merged_paths)

    def __iadd__(self, other_env: 'FSEnvironment') -> 'FSEnvironment':
        self = Environment.merge(self, other_env)
        return self
