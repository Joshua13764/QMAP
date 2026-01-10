from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from shutil import rmtree
from typing import Callable, List, Tuple

from joblib import delayed
from polars import LazyFrame, concat
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.polars_lazy_csv_adapter import \
    FSPolarsLazyCSVAdapter

LazyFrameAction = Callable[[], LazyFrame]
LazyFrameActionBatch = List[LazyFrameAction]


@dataclass(frozen=True, kw_only=True)
class FSPolarsLazyActionCSVBatched(
        FSAdapterBase[LazyFrameActionBatch, FSPathLocalDisk]):
    standard_extension: str | None | bool = field(default="csv")
    temp_folder_path: str
    lazy_frame_adapter: FSPolarsLazyCSVAdapter = field(
        default_factory=lambda: FSPolarsLazyCSVAdapter())
    temp_path_function: Callable[[LazyFrame, int], Tuple[str, ...]] = field(
        default=lambda obj, obj_index: ("temp", f"export obj {obj_index}"))
    lazy_merge_function: Callable[[List[LazyFrame]], LazyFrame] = field(
        default=lambda dfs: concat(dfs))
    n_jobs: int = field(default=4)

    def read(self, path: FSPathLocalDisk) -> LazyFrameActionBatch:
        raise NotImplementedError

    def write(self, obj: LazyFrameActionBatch, path: FSPathLocalDisk) -> None:
        self.clean_temp_folder()

        temp_file_paths: List[FSPathLocalDisk] = self.export_temp_files(
            action_batch=obj,
            root_path=path
        )

        temp_files: List[LazyFrame] = [
            FSObject(path, self.lazy_frame_adapter).object for path in temp_file_paths]

        merged_file: LazyFrame = self.lazy_merge_function(temp_files)

        FSEnvironment.save(merged_file, path, self.lazy_frame_adapter)

        self.clean_temp_folder()

    def clean_temp_folder(self) -> None:
        Path(self.temp_folder_path).mkdir(parents=True, exist_ok=True)
        rmtree(self.temp_folder_path)
        Path(self.temp_folder_path).mkdir(parents=True, exist_ok=True)

    def export_temp_files(
            self, action_batch: LazyFrameActionBatch, root_path: FSPathLocalDisk) -> List[FSPathLocalDisk]:

        message: str = f"Saving LazyFrame action with {self.n_jobs} jobs"
        unit: str = "sub frame"

        # parallel_results_raw: List[FSPathLocalDisk] = [
        #     FSPolarsLazyActionCSVBatched.export_obj(
        #         action,
        #         self.lazy_frame_adapter,
        #         lambda obj: FSPolarsLazyActionCSVBatched.get_temp_path(
        #             self.temp_path_function, self.temp_folder_path, obj, action_index)
        #     )
        #     for action_index, action in enumerate(action_batch)
        # ]

        parallel_results_raw = ParallelPbar(message, unit=unit)(n_jobs=self.n_jobs)(
            delayed(FSPolarsLazyActionCSVBatched.export_obj)(
                action,
                self.lazy_frame_adapter,
                partial(
                    FSPolarsLazyActionCSVBatched.get_temp_path,
                    self.temp_path_function,
                    self.temp_folder_path,
                    obj_index=action_index,
                )
            )
            for action_index, action in enumerate(action_batch)
        )

        assert all(
            parallel_result_raw is not None for parallel_result_raw in parallel_results_raw)

        parallel_results_cleaned: List[FSPathLocalDisk] = [
            parallel_result_raw for parallel_result_raw in parallel_results_raw if parallel_result_raw is not None]

        return parallel_results_cleaned

    @staticmethod
    def export_obj(
            obj_action: Callable[[], LazyFrame], adapter: FSAdapterBase[LazyFrame, FSPathLocalDisk],
            temp_path_function_partial: Callable[[LazyFrame], FSPathLocalDisk]) -> FSPathLocalDisk:

        obj: LazyFrame = obj_action()
        temp_path: FSPathLocalDisk = temp_path_function_partial(obj)

        FSEnvironment.save(
            obj=obj_action(),
            path=temp_path,
            adapter=adapter)

        return temp_path

    @staticmethod
    def get_temp_path(temp_path_function: Callable[[LazyFrame, int], Tuple[str, ...]], temp_folder: str,
                      obj: LazyFrame, obj_index: int) -> FSPathLocalDisk:
        return FSPathLocalDisk(
            root_path=temp_folder,
            path=temp_path_function(obj, obj_index),
            markers=tuple()
        )
