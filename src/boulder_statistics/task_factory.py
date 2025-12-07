from typing import Callable

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.step_base import StepBase
from boulder_statistics.task_step_base import TaskStepBase


class TaskFactory():

    @staticmethod
    def construct_task(
            step: StepBase, result_cache: ResultCache[StepBase, FSEnvironment]) -> Callable[[FSEnvironment], FSEnvironment]:
        match step:
            case TaskStepBase(): return TaskFactory.handle_task_step_base(step, result_cache)
            case _: raise TypeError(f"Step {step} has no task factory handler")

    @staticmethod
    def handle_task_step_base(
            step: TaskStepBase, result_cache: ResultCache[StepBase, FSEnvironment]) -> Callable[[FSEnvironment], FSEnvironment]:

        print(f"Compiling task {step.task_name}...")

        if result_cache.does_result_cache_exist(
                step, save_prefix=step.task_name):

            def load_cache_step_task(env: FSEnvironment) -> FSEnvironment:
                res: FSEnvironment = result_cache.open_result_cache(
                    step, save_prefix=step.task_name)

                print(f"Running task {step.task_name} from cache...")

                return res

            return load_cache_step_task
        else:
            def run_step_task(env: FSEnvironment) -> FSEnvironment:
                print(f"Running task {step.task_name}...")
                res: FSEnvironment = step.run(env)

                print(f"Caching task {step.task_name} result...")
                result_cache.save_result_cache(
                    step, save_prefix=step.task_name, result_cache=res)

                return res

            return run_step_task
