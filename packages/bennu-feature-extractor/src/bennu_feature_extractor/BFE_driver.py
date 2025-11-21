from graphlib import TopologicalSorter
from typing import Any, Coroutine, List

from prefect import flow, task
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase


class StepsOrchestrator:
    @staticmethod
    def run_steps(tasks: List[StepBase], result_cache: LocalFileSystem | Coroutine[Any, Any, LocalFileSystem],
                  flow_name: str = "Run all steps in auto DAG") -> dict[str, PrefectFuture[FSEnvironment]]:

        step_order: List[StepBase] = StepsOrchestrator.get_step_order(tasks)
        return flow(name=flow_name)(StepsOrchestrator.compile_steps)(
            step_order, result_cache)

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
            step_order: List[StepBase], result_cache: LocalFileSystem | Coroutine[Any, Any, LocalFileSystem]):

        future_results: dict[str, PrefectFuture[FSEnvironment]] = {}

        for step in step_order:

            step_required_upstream_futures: List[PrefectFuture[FSEnvironment]] = [
                future_results[n] for n in step.run_after_task_names]

            compiled_task: PrefectFuture[FSEnvironment] = step.get_task(
                result_cache).submit(
                    StepsOrchestrator.handle_env_merging(
                        step_required_upstream_futures),
                    step)

            future_results[step.task_name] = compiled_task

        return future_results

    @staticmethod
    def handle_env_merging(
            step_required_upstream_futures: List[PrefectFuture[FSEnvironment]]) -> PrefectFuture[FSEnvironment]:
        match len(step_required_upstream_futures):
            case 0: return task(name="create_env")(FSEnvironment.empty).submit()
            case 1: return step_required_upstream_futures[0]
            case _: return task(name="merge_envs")(FSEnvironment.merge).submit(step_required_upstream_futures)
