from dataclasses import dataclass, field
from pathlib import Path
from sys import prefix
from typing import Callable, List

from joblib import delayed
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from bennu_feature_extractor.environment_tools.file_storage_adapters.copy_adapter import \
    FSShutilCopy2Adapter
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from bennu_feature_extractor.task_step_base import TaskStepBase


@dataclass(frozen=True)
class SimpleFunctionImportExport[T](TaskStepBase):
    adapter: FSAdapterBase[T, FSPathLocalDisk]
    input_markers: frozenset[FSMarkerBase]
    output_markers: frozenset[FSMarkerBase]
    function_to_apply: Callable[[T], T] = field(
        hash=False, repr=False, compare=False)
    output_name_prefix: str = field(default_factory=lambda: "")
    output_name_suffix: str = field(default_factory=lambda: "")

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_apply_to: List[FSPathLocalDisk] = env.get_paths_from_markers(
            FSPathLocalDisk, self.input_markers)

        export_files: List[FSPathLocalDisk] = [file.copy_with_stem_prefix_and_suffix(
            stem_prefix=self.output_name_prefix, stem_suffix=self.output_name_prefix, markers=self.output_markers)
            for file in files_to_apply_to]

        run_for_obj: Callable[[FSPathLocalDisk, FSPathLocalDisk], FSPathBase] = lambda in_path, out_path: FSEnvironment.save(
            self.function_to_apply(FSEnvironment.load(in_path, self.adapter)),
            path=out_path,
            adapter=self.adapter,
        )

        [
            run_for_obj(in_path, out_path)
            for in_path, out_path in zip(files_to_apply_to, export_files)
            if out_path.exists == False
        ]

        # ParallelPbar(
        #     f"Processing objects for task {self.task_name}", unit="object")(n_jobs=-1)(
        #     delayed(run_for_obj)(in_path, out_path)
        #     for in_path, out_path in zip(files_to_apply_to, export_files)
        # )

        return FSEnvironment(paths=frozenset(export_files))
