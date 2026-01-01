from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path
from sys import stdout
from typing import Any, Callable, Dict, List

from stablehash import stablehash

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
    hash_method: Callable[[Any], str] = field(
        default=lambda obj: stablehash(obj).hexdigest())

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

    @staticmethod
    def get_task_names(*tasks) -> set[str]:
        return set(task.task_name for task in tasks)

    @property
    def task_hash(self) -> str:
        return self.hash_method(
            stablehash(self.cleaned_hashable))

    @property
    def cleaned_hashable(self) -> tuple[Any, ...]:
        return tuple(
            HashCleanerFactory.clean_hashable(i)
            for i in self.hashable
        )

    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...

    @property
    @abstractmethod
    def hashable(self) -> tuple[Any, ...]:
        ...
