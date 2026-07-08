import shutil
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from polars import DataFrame, LazyFrame
from tqdm import tqdm
from tqdm_joblib import ParallelPbar

from boulder_statistics.analysis.data_product_encyclopedia import FACES
from boulder_statistics.refinement_plus.bounding_box import BoundingBox
from boulder_statistics.refinement_plus.qcube_chunk import QCubeChunk


class ChunkingTools:
    @staticmethod
    def get_chunk_bbox(
        lf: LazyFrame,
        chunk: QCubeChunk,
        xyz_columns: list[str] = ["x", "y", "z"],
    ) -> BoundingBox:
        lf_filtered: LazyFrame = chunk.filter_lf(lf)

        x, y, z = xyz_columns

        result = (
            lf_filtered
            .select(
                pl.col(x).min().alias("x_min"),
                pl.col(x).max().alias("x_max"),
                pl.col(y).min().alias("y_min"),
                pl.col(y).max().alias("y_max"),
                pl.col(z).min().alias("z_min"),
                pl.col(z).max().alias("z_max"),
            )
            .collect()
            .row(0)
        )

        return BoundingBox(
            x_min=np.float64(result[0]),
            x_max=np.float64(result[1]),
            y_min=np.float64(result[2]),
            y_max=np.float64(result[3]),
            z_min=np.float64(result[4]),
            z_max=np.float64(result[5]),
        )

    @staticmethod
    def extract_chunks(lf: LazyFrame, chunk: QCubeChunk,
                       columns: List[str], numb_workers: int = 4,
                       verbose=False) -> List[np.ndarray]:

        lf_filtered: LazyFrame = chunk.filter_lf(lf)

        def extract_func(chunk_lf: LazyFrame, column_name: str) -> np.ndarray:
            col_data: pl.DataFrame = chunk_lf.select(
                column_name, "i", "j").collect()

            values = col_data[column_name].to_numpy()
            i = col_data["i"].to_numpy()
            j = col_data["j"].to_numpy()

            arr = np.empty(
                (i.max() - i.min() + 1,
                 j.max() - j.min() + 1),
                dtype=values.dtype)
            arr[i - i.min(), j - j.min()] = values

            return arr

        parallel = (
            ParallelPbar(
                f"Extracting from chunk {chunk} into columns {columns}"
            )(n_jobs=numb_workers)
            if verbose
            else Parallel(n_jobs=numb_workers)
        )

        extract_res = parallel(
            delayed(extract_func)(lf_filtered, column)
            for column in columns
        )

        return list(extract_res)

    @staticmethod
    def append_by_chunks(
            target_lf: LazyFrame, export_folder: Path, col_name: str,
            process_chunk: Callable[[QCubeChunk], np.ndarray],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=1),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk_lf: LazyFrame = chunk.filter_lf(target_lf)

            arr: np.ndarray = process_chunk(chunk)

            chunk_df: DataFrame = chunk_lf.collect()

            values: np.ndarray = arr[
                chunk_df["i"].to_numpy() - chunk.i_min,
                chunk_df["j"].to_numpy() - chunk.j_min
            ]

            chunk_df = chunk_df.with_columns(
                pl.Series(col_name, values)
            )

            chunk_df.write_parquet(
                export_folder / f"{chunk.short_name}.parquet"
            )

    @staticmethod
    def bulk_append_by_chunks(
            target_lf: LazyFrame, export_folder: Path, col_names: List[str],
            process_chunk: Callable[[QCubeChunk], List[np.ndarray]],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=1),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk_lf: LazyFrame = chunk.filter_lf(target_lf)
            chunk_df: DataFrame = chunk_lf.collect()

            for arr, col_name in zip(process_chunk(chunk), col_names):

                values: np.ndarray = arr[
                    chunk_df["i"].to_numpy() - chunk.i_min,
                    chunk_df["j"].to_numpy() - chunk.j_min
                ]

                chunk_df = chunk_df.with_columns(
                    pl.Series(col_name, values)
                )

            chunk_df.write_parquet(
                export_folder / f"{chunk.short_name}.parquet"
            )
