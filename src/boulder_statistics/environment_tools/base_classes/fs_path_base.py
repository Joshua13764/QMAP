from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Tuple

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase


@dataclass(frozen=True)
class FSPathBase(ABC):
    path: Tuple[str, ...]
    markers: tuple[FSMarkerBase, ...]

    @property
    def markers_lookup(self) -> Set[FSMarkerBase]:
        return set(self.markers)

    @abstractmethod
    def make_directory(self) -> None:
        ...

    @property
    @abstractmethod
    def exists(self) -> bool:
        ...
