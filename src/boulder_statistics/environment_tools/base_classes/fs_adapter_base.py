from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@dataclass(frozen=True)
class FSAdapterBase[ObjType, PathType: FSPathBase](ABC):
    @abstractmethod
    def write(self, obj: ObjType, path: PathType) -> None:
        ...

    @abstractmethod
    def read(self, path: PathType) -> ObjType:
        ...
