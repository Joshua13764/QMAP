from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path
from sys import stdout
from typing import Any, Dict, List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.hash_cleaner_factory import HashCleanerFactory

basicConfig(
    stream=stdout,
    level=INFO,
    format="%(levelname)s:%(name)s: %(message)s")


@dataclass(frozen=True, kw_only=True)
class StepBase(ABC):
    task_name: str
    ignore_cache: bool = field(default_factory=lambda: False, repr=False)
    run_after_task_names: tuple[str, ...] = field(
        default=(), repr=True)
    task_description: str = field(
        default="No task description provided", repr=True)

    # When the task version changes the the older cache will be invalid
    # (should override)
    @property
    def task_version(self) -> str: return "0.0.0"

    @property
    def logger(self) -> Logger:
        return getLogger(self.task_name)

    def get_dependencies(
            self, dependency_pool: Dict[str, "StepBase"]) -> List["StepBase"]:

        return [self] + [dependency for name in self.run_after_task_names for dependency in [
            *dependency_pool[name].get_dependencies(dependency_pool)]]

    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...

    @staticmethod
    def get_task_names(*tasks) -> set[str]:
        return set(task.task_name for task in tasks)

    @property
    def cleaned_hashable(self) -> tuple[Any, ...]:
        return tuple(
            HashCleanerFactory.clean_hashable(i)
            for i in self.hashable
        )

    @property
    @abstractmethod
    def hashable(self) -> tuple[Any, ...]:
        ...
