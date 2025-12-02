from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class FSMarkerBase(ABC):
    @abstractmethod
    def is_equivalent(self, target: 'FSMarkerBase') -> bool:
        ...
