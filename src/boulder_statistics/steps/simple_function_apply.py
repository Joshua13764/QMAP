
from dataclasses import dataclass, field
from typing import Any, Callable, List

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.step_default_markers import StepDefaultMarkers
from boulder_statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class SimpleFunctionApply[T](TaskStepBase, StepDefaultMarkers):
    read_adapter: FSAdapterBase[T, FSPathLocalDisk]
    write_adapter: FSAdapterBase[T, FSPathLocalDisk]
    import_folder: FSPathLocalDisk
    export_folder: FSPathLocalDisk
    function_to_apply: Callable[[T], T] = field(
        hash=False, repr=False, compare=False)
    output_name_prefix_no_extension: str = field(default="")
    output_name_suffix_no_extension: str = field(default="")
    n_jobs: int = field(default=1)

    @property
    def hashable(self) -> tuple[Any, ...]:
        return self.include_markers_in_hashable(
            self.read_adapter, self.write_adapter, self.import_folder,
            self.export_folder, self.output_name_prefix_no_extension,
            self.output_name_suffix_no_extension)

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_apply_to: List[FSPathLocalDisk] = self.get_files_with_markers(
            env)

        files_to_apply_to_stem_changes: List[FSPathLocalDisk] = [file.copy_with_stem_prefix_and_suffix(
            stem_prefix=self.output_name_prefix_no_extension, stem_suffix=self.output_name_suffix_no_extension, markers=self.output_markers)
            for file in files_to_apply_to]

        export_files: List[FSPathLocalDisk] = [
            file.transfer_root(
                self.import_folder,
                self.export_folder,
                markers=self.output_markers)
            for file in files_to_apply_to_stem_changes
        ]

        run_for_obj: Callable[[FSPathLocalDisk, FSPathLocalDisk], FSPathBase] = SimpleFunctionApply.get_apply_action(
            function=self.function_to_apply,
            read_adapter=self.read_adapter,
            write_adapter=self.write_adapter,
        )

        run_actions: List[Callable[[], FSPathBase]] = [
            lambda: run_for_obj(in_path, out_path)
            for in_path, out_path in zip(files_to_apply_to, export_files)
            if out_path.exists == False
        ]

        self.run_actions_in_parallel(
            run_actions, message="Running simple function apply actions in parallel", unit="file", n_jobs=self.n_jobs)

        return FSEnvironment(paths=tuple(export_files))

    @staticmethod
    def get_apply_action(function: Callable[[T], T], read_adapter: FSAdapterBase[T, FSPathLocalDisk],
                         write_adapter: FSAdapterBase[T, FSPathLocalDisk]
                         ) -> Callable[[FSPathLocalDisk, FSPathLocalDisk], FSPathBase]:

        return lambda in_path, out_path: FSEnvironment.save(
            obj=function(
                FSEnvironment.load(
                    in_path, read_adapter
                )
            ),
            path=out_path,
            adapter=write_adapter,
        )
