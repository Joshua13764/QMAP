from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from prefect.tasks import Task
from functools import cached_property

from .environment import Environment

@dataclass
class StepBase(ABC):
    _logger: Logger

    @cached_property
    def task(self) -> Task:
        return Task(
            fn=self.run,
            name=f"{type(self).__name__}.run",
            task_run_name="{env}",
        )

    @abstractmethod
    def run(self, env: Environment) -> Environment:
        ...