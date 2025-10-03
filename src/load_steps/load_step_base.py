from ..pipeline_step_base import PipelineStepBase

from abc import abstractmethod

class LoadStepBase(PipelineStepBase):

    @property
    def step_type(self) -> str:
        return "load"
    
    @abstractmethod
    def run(self):
        pass