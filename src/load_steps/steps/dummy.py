from ..load_step_base import LoadStepBase

class DummyLoadStep(LoadStepBase):
    @property
    def name(self) -> str:
        return "dummy_load_step"
    
    def run(self):
        print("Dummy load step executed.")