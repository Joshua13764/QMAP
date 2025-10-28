from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from prefect import task
from prefect.results import ResultStorage
from prefect.serializers import PickleSerializer
from prefect import get_run_logger
from bennu_feature_extractor.environment import Environment

@dataclass
class StepBase(ABC):
    result_storage: ResultStorage

    @property
    def logger(self) -> Logger:
        return get_run_logger()

    @property
    def get_task(self):

        @task(
            result_storage=self.result_storage,
            result_serializer=PickleSerializer(),
            cache_key_fn = lambda context, cfg : str((cfg["env"].get_cache_key(), self.get_hash()).__hash__()),
        )

        def _step_task(env: Environment) -> Environment:
            return self.run(env)

        return _step_task
    
    @property
    def get_task_no_cache(self):
        @task
        def _step_task_no_cache(env: Environment) -> Environment:
            return self.run(env)
        
        return _step_task_no_cache

    @abstractmethod
    def get_hash(self) -> int:
        ...

    @abstractmethod
    def run(self, env: Environment) -> Environment:
        ...

    

    