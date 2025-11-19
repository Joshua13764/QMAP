from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Coroutine, FrozenSet, Iterable, List, Mapping, Sequence

from prefect import flow, task
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor.step_templates.simple_request import SimpleRequest


class BFEDriver:
    @staticmethod
    def run_steps(tasks: List[StepBase], result_cache: LocalFileSystem | Coroutine[Any, Any, LocalFileSystem],
                  flow_name: str = "Run all steps in auto DAG") -> dict[str, PrefectFuture[FSEnvironment]]:

        dag_flow = flow(name=flow_name)(run_step_dag)
        return dag_flow(tasks, result_cache)


@task(name="merge_envs")
def merge_envs(
    base_env: "FSEnvironment",
    upstream_envs: list["FSEnvironment"],
) -> "FSEnvironment":
    return FSEnvironment.merge(
        upstream_envs +
        [base_env])


def run_step_dag(
        steps: List[StepBase], result_storage) -> dict[str, PrefectFuture[FSEnvironment]]:
    base_env: FSEnvironment = FSEnvironment.empty()
    name_to_step: dict[str, StepBase] = {s.task_name: s for s in steps}
    graph: dict[str, set[str]] = {s.task_name: set(
        s.run_after_task_names) for s in steps}
    ts: TopologicalSorter[str] = TopologicalSorter(graph)
    order: List[str] = list(ts.static_order())

    futures: dict[str, PrefectFuture[FSEnvironment]] = {}

    for task_name in order:
        step: StepBase = name_to_step[task_name]
        upstream_names: FrozenSet[str] = step.run_after_task_names
        upstream_futures: List[PrefectFuture[FSEnvironment]] = [
            futures[n] for n in upstream_names]

        if upstream_futures:
            merged_env_future: PrefectFuture[FSEnvironment] = merge_envs.submit(
                base_env,
                upstream_futures,
            )
            env_arg: PrefectFuture[FSEnvironment] = merged_env_future
        else:
            env_arg = base_env

        compiled_task = step.get_task(result_storage)

        future: PrefectFuture[FSEnvironment] = compiled_task.submit(
            env_arg,
            step,
        )

        futures[task_name] = future

    return futures
