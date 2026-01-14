from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List

from attr import field
from joblib import delayed
from tenacity import retry, stop_after_attempt
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.step_base import StepBase


@dataclass(frozen=True, kw_only=True)
class TaskStepBase(StepBase):

    @abstractmethod
    def run(self, env: FSEnvironment) -> FSEnvironment:
        ...

    def run_actions_in_parallel[O](self, functions: List[Callable[[], O]],
                                   message: str = "", n_jobs: int = -1, unit: str = "") -> List[O]:

        return self.run_in_parallel(
            function=lambda func: func(),
            inputs=functions,
            message=message,
            n_jobs=n_jobs,
            unit=unit,
        )

    def run_in_parallel[I, O](
            self, function: Callable[[I], O], inputs: List[I],
            message: str = "", n_jobs: int = -1, unit: str = "") -> List[O]:

        if self.debug_mode or len(inputs) == 1:
            print(message)
            return [function(input) for input in inputs]

        parallel_results_raw = ParallelPbar(message, unit=unit)(n_jobs=n_jobs)(
            delayed(function)(input)
            for input in inputs
        )

        assert all(
            parallel_result_raw is not None for parallel_result_raw in parallel_results_raw)

        parallel_results_cleaned: List[O] = [
            parallel_result_raw for parallel_result_raw in parallel_results_raw if parallel_result_raw is not None]

        return parallel_results_cleaned
