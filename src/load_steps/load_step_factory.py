from .steps.dummy import DummyLoadStep

class LoadStepFactory:
    @staticmethod
    def create_dummy_load_pipeline_step() -> DummyLoadStep:
        return DummyLoadStep()