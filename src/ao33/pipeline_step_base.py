from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger

@dataclass
class PipelineStepBase(ABC):
    _logger : Logger

    @property
    @abstractmethod
    def step_type(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self) -> None:
        pass