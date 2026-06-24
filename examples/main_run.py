from pathlib import Path
from typing import List

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.refinement import refinement_settings
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.standard_infer_run import StandardInferRun
from boulder_statistics.standard_refine_run import StandardRefineRun
from boulder_statistics.step_base import StepBase
from boulder_statistics.steps_orchestrator import StepsOrchestrator

cache_folder = Path(r"G:\AO33Cont_cache")

pipeline_cache_path: Path = cache_folder / "pipeline_cache"
data_pack_export_folder: Path = cache_folder / Path("data_pack")

# TODO
# refinement_settings = refinement_settings.RefinementSettings(
#     data_products_path=cache_folder / "data_products"
#     figures_path=cache_folder / "figures",
#     raw_database_file_path=cache_folder /,  # TODO
#     refinement_cache_path=cache_folder / "refinement_cache"
# )

if __name__ == "__main__":
    pass
    # ~ 100 GB of data so you will need to account for that!

    # May take several hours (needs dedicated GPU for AI inference)
    # steps: List[StepBase] = StandardInferRun.get_infer_steps(
    #     pipeline_cache_path, data_pack_export_folder)
    # cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
    #     cache_folder=Path(".cache"), result_type=FSEnvironment)
    # futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
    #     steps, cache)

    # # Will take an hour and ~ 50 GB to run refinement
    # df, df_meta = StandardRefineRun.step_0(refinement_settings)
    # StandardRefineRun.step_1(refinement_settings, df, df_meta)
    # StandardRefineRun.step_2(refinement_settings, df_meta)
    # df = StandardRefineRun.step_3_1(refinement_settings, df)
    # df = StandardRefineRun.step_3_2(refinement_settings, df)
    # StandardRefineRun.step_3_3(refinement_settings, df_meta)
    # StandardRefineRun.step_3_4(refinement_settings)
