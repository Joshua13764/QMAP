from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from attr import field

from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@dataclass(frozen=True, kw_only=True)
class FSAdapterBase[ObjType, PathType: FSPathBase](ABC):
    # str will add ".{str}"
    # None will throw error if extensionless file passed
    # bool false will not add extension
    standard_extension: str | None | bool = field(default=None)

    @abstractmethod
    def write(self, obj: ObjType, path: PathType) -> None:
        ...

    @abstractmethod
    def read(self, path: PathType) -> ObjType:
        ...
