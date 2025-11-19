from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from typing import FrozenSet, List

from prefect import get_run_logger, task
from prefect.cache_policies import INPUTS
from prefect.futures import PrefectFuture
from prefect.results import ResultStorage
from prefect.serializers import PickleSerializer

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment


@dataclass(frozen=True, kw_only=True)
class StepBase(ABC):
    task_name: str
    run_after_task_names: FrozenSet[str]
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

    def get_task(self, result_storage: ResultStorage):

        @task(name=self.task_name, description=self.task_description,
              result_storage=result_storage, persist_result=self.persist_result,
              cache_policy=INPUTS)
        def compiled_task(env: FSEnvironment, step: StepBase) -> FSEnvironment:
            return step.run(env)

        return compiled_task

    def submit_task(self, result_storage: ResultStorage, env: FSEnvironment = FSEnvironment.empty(),
                    ) -> PrefectFuture[FSEnvironment]:
        return self.get_task(result_storage).submit(env)

    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...
