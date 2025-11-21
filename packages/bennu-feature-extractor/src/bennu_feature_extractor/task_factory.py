from prefect import task
from prefect.cache_policies import INPUTS
from prefect.results import ResultStorage
from prefect.tasks import Task

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor.task_step_base import TaskStepBase


class TaskFactory():

    @staticmethod
    def construct_task(step: StepBase, result_storage: ResultStorage |
                       None) -> Task[..., FSEnvironment]:
        match step:
            case TaskStepBase(): return TaskFactory.handle_task_step_base(step, result_storage)
            case _: raise TypeError(f"Step {step} has no task factory handler")

    @staticmethod
    def handle_task_step_base(
            step: TaskStepBase, result_storage: ResultStorage | None) -> Task[..., FSEnvironment]:

        @task(
            name=step.task_name,
            description=step.task_description,
            result_storage=result_storage,
            persist_result=step.persist_result,
            cache_policy=INPUTS,
        )
        def wrapped_task(env: FSEnvironment,
                         step: 'StepBase') -> FSEnvironment:
            return step.run(env)

        return wrapped_task
