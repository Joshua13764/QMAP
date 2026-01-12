from dataclasses import dataclass, field
from functools import partial
from math import floor
from pathlib import Path
from shutil import rmtree
from typing import Callable, List, Tuple

import polars as pl
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
from boulder_statistics.file_storage_adapters.polars_lazy_parquet_adapter import \
    FSPolarsLazyParquetAdapter
from boulder_statistics.file_storage_adapters.polars_lazy_parquet_adapter_large_export import \
    FSPolarsLazyParquetAdapterLargeExport

LazyFrameAction = Callable[[], LazyFrame]
LazyFrameActionBatch = List[LazyFrameAction]


@dataclass(frozen=True, kw_only=True)
class FSPolarsLazyActionBatchedSmartExport(
        FSAdapterBase[LazyFrameActionBatch, FSPathLocalDisk]):
    standard_extension: str | None | bool = field(default="parquet")
    temp_folder_path: str
    temp_lazy_frame_adapter: FSAdapterBase[LazyFrame, FSPathLocalDisk] = field(
        default_factory=lambda: FSPolarsLazyParquetAdapter())
    temp_path_function: Callable[[LazyFrame, int], Tuple[str, ...]] = field(
        default=lambda obj, obj_index: ("temp", f"export obj {str(obj_index).zfill(9)}"))
    lazy_merge_function: Callable[[List[LazyFrame]], LazyFrame] = field(
        default=lambda dfs: concat(dfs))
    n_jobs: int = field(default=4)

    def read(self, path: FSPathLocalDisk) -> LazyFrameActionBatch:
        raise NotImplementedError

    def write(self, obj: LazyFrameActionBatch, path: FSPathLocalDisk) -> None:
        self.clean_temp_folder()

        temp_file_paths, row_count = self.export_temp_files(
            action_batch=obj,
            root_path=path
        )

        temp_files: List[LazyFrame] = [
            FSObject(path, self.temp_lazy_frame_adapter).object for path in temp_file_paths]

        merged_file: LazyFrame = self.lazy_merge_function(temp_files)

        # Aim for ~ 30 MB chunks
        batches_to_export: float = self.get_df_parts_export_size_mb() / 30
        rows_per_batch: int = floor(row_count / batches_to_export)

        print(
            f"""Merging and exporting data in ~ {
                int(batches_to_export)} batches""")

        FSEnvironment.save(
            merged_file,
            path,
            FSPolarsLazyParquetAdapterLargeExport(
                row_group_size=rows_per_batch
            )
        )

        self.clean_temp_folder()

    def get_df_parts_export_size_mb(self) -> float:
        return sum([f.stat().st_size for f in Path(
            self.temp_folder_path).glob("*.parquet")]) / 1_048_576

    def clean_temp_folder(self) -> None:
        Path(self.temp_folder_path).mkdir(parents=True, exist_ok=True)
        rmtree(self.temp_folder_path)
        Path(self.temp_folder_path).mkdir(parents=True, exist_ok=True)

    def export_temp_files(
            self, action_batch: LazyFrameActionBatch, root_path: FSPathLocalDisk) -> Tuple[List[FSPathLocalDisk], int]:

        message: str = f"Saving LazyFrame action with {self.n_jobs} jobs"
        unit: str = "sub frame"

        parallel_results_raw = ParallelPbar(message, unit=unit)(n_jobs=self.n_jobs)(
            delayed(FSPolarsLazyActionBatchedSmartExport.export_obj)(
                action,
                self.temp_lazy_frame_adapter,
                partial(
                    FSPolarsLazyActionBatchedSmartExport.get_temp_path,
                    self.temp_path_function,
                    self.temp_folder_path,
                    obj_index=action_index,
                )
            )
            for action_index, action in enumerate(action_batch)
        )

        assert all(
            parallel_result_raw is not None for parallel_result_raw in parallel_results_raw)

        parallel_results_cleaned: List[Tuple[FSPathLocalDisk, int]] = [
            parallel_result_raw for parallel_result_raw in parallel_results_raw if parallel_result_raw is not None]

        return [export_data[0] for export_data in parallel_results_cleaned], sum(
            [export_data[1] for export_data in parallel_results_cleaned])

    @staticmethod
    def export_obj(
            obj_action: Callable[[], LazyFrame], adapter: FSAdapterBase[LazyFrame, FSPathLocalDisk],
            temp_path_function_partial: Callable[[LazyFrame], FSPathLocalDisk]) -> Tuple[FSPathLocalDisk, int]:

        obj: LazyFrame = obj_action()
        temp_path: FSPathLocalDisk = temp_path_function_partial(obj)

        row_count: int = obj.select(pl.len()).collect().item()

        FSEnvironment.save(
            obj=obj_action(),
            path=temp_path,
            adapter=adapter)

        return temp_path, row_count

    @staticmethod
    def get_temp_path(temp_path_function: Callable[[LazyFrame, int], Tuple[str, ...]], temp_folder: str,
                      obj: LazyFrame, obj_index: int) -> FSPathLocalDisk:
        return FSPathLocalDisk(
            root_path=temp_folder,
            path=temp_path_function(obj, obj_index),
            markers=tuple()
        )
