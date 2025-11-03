from abc import ABC, abstractmethod

import attr


@attr.define(frozen=True, slots=True)
class FSMarkerBase(ABC):
    @abstractmethod
    def is_equivalent(self, target: 'FSMarkerBase') -> bool:
        ...
