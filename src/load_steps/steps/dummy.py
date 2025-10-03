from ..load_step_base import LoadPipelineStepBase

class DummyLoadStep(LoadPipelineStepBase):
    @property
    def name(self) -> str:
        return "dummy_load_step"
    
    def run(self):
        print("Dummy load step executed.")