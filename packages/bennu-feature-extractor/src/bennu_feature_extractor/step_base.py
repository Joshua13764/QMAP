from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, FrozenSet, List

from prefect import get_run_logger

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment


@dataclass(frozen=True, kw_only=True)
class StepBase(ABC):
    task_name: str
    run_after_task_names: FrozenSet[str] = field(
        default=frozenset(), repr=True)
    task_description: str = field(
        default="No task description provided", repr=True)
    persist_result: bool = field(default=True, repr=True)

    # When the task version changes the the older cache will be invalid
    # (should override)
    @property
    def task_version(self) -> str: return "0.0.0"

    @property
    def logger(self) -> Logger:
        return get_run_logger()

    def get_dependencies(
            self, dependency_pool: Dict[str, "StepBase"]) -> List["StepBase"]:

        return [self] + [dependency for name in self.run_after_task_names for dependency in [
            *dependency_pool[name].get_dependencies(dependency_pool)]]

    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...

    @staticmethod
    def get_task_names(*tasks) -> frozenset[str]:
        return frozenset([task.task_name for task in tasks])
