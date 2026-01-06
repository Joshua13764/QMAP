from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@dataclass(frozen=True)
class FSAdapterBase[ObjType, PathType: FSPathBase](ABC):
    # str will add ".{str}"
    # None will throw error if extensionless file passed
    # bool false will not add extension
    standard_extension: ClassVar[str | None | bool] = None

    @abstractmethod
    def write(self, obj: ObjType, path: PathType) -> None:
        ...

    @abstractmethod
    def read(self, path: PathType) -> ObjType:
        ...
