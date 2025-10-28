from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from prefect import task

from .environment import Environment

@dataclass
class StepBase(ABC):
    logger: Environment

    @property
    def get_task(self):
        @task(persist_result=True)
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
    def run(self, env: Environment) -> Environment:
        ...

    

    