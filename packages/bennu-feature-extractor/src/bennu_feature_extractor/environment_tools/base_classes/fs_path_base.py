from abc import ABC, abstractmethod
from typing import Tuple

import attr

from bennu_feature_extractor.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase


@attr.define(frozen=True, slots=True)
class FSPathBase(ABC):
    path: Tuple[str, ...]
    markers: frozenset[FSMarkerBase]

    @abstractmethod
    def make_directory(self) -> None:
        ...
