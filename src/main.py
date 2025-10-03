from .load_steps.load_step_factory import LoadStepFactory
from .load_steps.load_step_base import LoadStepBase

if __name__ == "__main__":
    load_step : LoadStepBase = LoadStepFactory.create_dummy_load_pipeline_step()
    load_step.run()
