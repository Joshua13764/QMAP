from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase


@dataclass(frozen=True)
class FSPathBase(ABC):
    path: Tuple[str, ...]
    markers: frozenset[FSMarkerBase]

    @abstractmethod
    def make_directory(self) -> None:
        ...
