from abc import abstractmethod
from dataclasses import dataclass

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.step_base import StepBase


@dataclass(frozen=True, kw_only=True)
class TaskStepBase(StepBase):
    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...
