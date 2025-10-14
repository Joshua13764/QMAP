from ..pipeline_step_base import PipelineStepBase

from abc import abstractmethod

class BoulderInferenceStepBase(PipelineStepBase):
    @property
    def step_type(self) -> str:
        return "BoulderInference"
    
    @abstractmethod
    def run(self):
        pass