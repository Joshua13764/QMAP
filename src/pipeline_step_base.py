from abc import ABC, abstractmethod

class PipelineStepBase(ABC):
    @property
    @abstractmethod
    def step_type(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self):
        pass