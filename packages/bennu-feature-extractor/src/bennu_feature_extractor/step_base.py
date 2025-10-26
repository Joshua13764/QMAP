from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from prefect import task

from .environment import Environment

@dataclass
class StepBase(ABC):
    _logger: Logger

    def get_task(self, env: Environment):
        @task
        def _inner_step_task() -> Environment:
            return self.run(env)

        return _inner_step_task

    @abstractmethod
    def run(self, env: Environment) -> Environment:
        ...