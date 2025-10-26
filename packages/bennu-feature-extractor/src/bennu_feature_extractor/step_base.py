from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from prefect import task

from .environment import Environment

@dataclass
class StepBase(ABC):
    _logger: Logger

    @property
    def get_task(self):
        @task
        def _step_task(env: Environment) -> Environment:
            return self.run(env)

        return _step_task

    @abstractmethod
    def run(self, env: Environment) -> Environment:
        ...