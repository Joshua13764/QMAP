from abc import abstractmethod
from dataclasses import dataclass

from Boulder_Statistics.environment_tools.fs_environment import FSEnvironment
from Boulder_Statistics.step_base import StepBase


@dataclass(frozen=True, kw_only=True)
class TaskStepBase(StepBase):
    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...
