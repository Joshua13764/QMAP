from pathlib import Path
from typing import List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.standard_run import StandardRun
from boulder_statistics.step_base import StepBase
from boulder_statistics.steps_orchestrator import StepsOrchestrator

if __name__ == "__main__":
    # ~ 100 GB of data so you will need to account for that!
    pipeline_cache_path: Path = Path(
        r"G:\AO33_pipeline_folders")
    data_pack_export_folder: Path = pipeline_cache_path / Path("data_pack")

    # May take several hours (needs dedicated GPU for AI inference)
    steps: List[StepBase] = StandardRun.get_standard_steps(
        pipeline_cache_path, data_pack_export_folder)
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)
    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
