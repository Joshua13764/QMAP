from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from prefect import task

from .environment import Environment

@dataclass
class StepBase(ABC):
    _logger: Logger

    @abstractmethod
    def run(self, env: Environment) -> Environment:
        ...