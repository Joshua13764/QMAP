from abc import abstractmethod
from dataclasses import dataclass

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase


@dataclass(frozen=True, kw_only=True)
class TaskStepBase(StepBase):
    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...
