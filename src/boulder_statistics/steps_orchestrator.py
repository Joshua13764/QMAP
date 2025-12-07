import inspect
from graphlib import TopologicalSorter
from typing import Callable, List, Set

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.step_base import StepBase
from boulder_statistics.task_factory import TaskFactory


class StepsOrchestrator:
    @staticmethod
    def run_steps(tasks: List[StepBase], result_cache: ResultCache[StepBase,
                  FSEnvironment]) -> dict[str, FSEnvironment]:

        step_order: List[StepBase] = StepsOrchestrator.get_step_order(tasks)

        return StepsOrchestrator.compile_steps(step_order, result_cache)

    @staticmethod
    def run_tasks_with_dependencies(tasks: List[StepBase], dependency_pool: List[StepBase],
                                    result_cache: ResultCache[StepBase, FSEnvironment]) -> dict[str, FSEnvironment]:

        dependency_pool_dict: dict[str, StepBase] = {
            dependency.task_name: dependency for dependency in dependency_pool}

        dependencies: Set[StepBase] = {dependency for task in tasks for dependency in task.get_dependencies(
            dependency_pool_dict)}

        print(
            f"Run tasks and dependencies resolved: {[d.task_name for d in dependencies]}")

        return StepsOrchestrator.run_steps(
            list(dependencies),
            result_cache)

    @staticmethod
    def get_step_order(steps: List[StepBase]) -> List[StepBase]:
        name_to_step: dict[str, StepBase] = {s.task_name: s for s in steps}

        graph: dict[str, set[str]] = {s.task_name: set(
            s.run_after_task_names) for s in steps}

        ts: TopologicalSorter[str] = TopologicalSorter(graph)
        order: List[str] = list(ts.static_order())

        return [name_to_step[step_name] for step_name in order]

    @staticmethod
    def compile_steps(
            step_order: List[StepBase], result_cache: ResultCache) -> dict[str, FSEnvironment]:

        future_results: dict[str, FSEnvironment] = {}

        for step in step_order:

            step_required_upstream_futures: List[FSEnvironment] = [
                future_results[n] for n in step.run_after_task_names]

            step_task: Callable[[FSEnvironment], FSEnvironment] = TaskFactory.construct_task(
                step, result_cache)

            input_environment: FSEnvironment = StepsOrchestrator.handle_env_merging(
                step_required_upstream_futures)

            submitted_task: FSEnvironment = step_task(input_environment)

            future_results[step.task_name] = submitted_task

        return future_results

    @staticmethod
    def handle_env_merging(
            step_required_upstream_futures: List[FSEnvironment]) -> FSEnvironment:
        match len(step_required_upstream_futures):
            case 0: return FSEnvironment.empty()
            case 1: return step_required_upstream_futures[0]
            case _:
                print("Merging environments...")
                return FSEnvironment.merge(step_required_upstream_futures)

    @staticmethod
    def auto_find_steps(frame=None) -> List[StepBase]:
        if frame is None:
            frame = inspect.currentframe().f_back

        namespace = {}
        namespace.update(frame.f_globals)
        namespace.update(frame.f_locals)

        result: List[StepBase] = [
            value
            for name, value in namespace.items()
            if isinstance(value, StepBase) and not isinstance(value, type)
        ]
        return result
