import shutil
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from polars import DataFrame, Expr, LazyFrame
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
    def join_full_with_aggs(
            export_folder: Path,
            full_db: pl.LazyFrame,
            aggs_to_join_with: Dict[Tuple[str], pl.DataFrame],  # Join on : df
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=3),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        def process_chunk(chunk: QCubeChunk) -> None:
            full_chunked_df: pl.LazyFrame = chunk.filter_lf(full_db)

            reduce(
                lambda left, right:
                    left.join(
                        right[1].lazy(),
                        on=list(right[0]),
                        how="left",
                        coalesce=True,
                    ),
                aggs_to_join_with.items(),
                full_chunked_df,
            ).sink_parquet(
                export_folder / f"{chunk.short_name}.parquet",
                engine="streaming"
            )

        for chunk in tqdm(chunks, desc="Joining full with aggs"):
            process_chunk(chunk)

    @staticmethod
    def join_full_with_agg(
            export_folder: Path,
            full_db: pl.LazyFrame,
            agg_db: pl.DataFrame,
            join_on: List[str] = ["boulder_id"],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=3),
            skip_if_exists=False, n_jobs=4) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        def process_chunk(chunk: QCubeChunk) -> None:
            full_chunked_df: DataFrame = chunk.filter_lf(full_db).collect()
            full_chunked_df.join(
                agg_db,
                on=join_on,
                how="inner"
            ).write_parquet(
                export_folder / f"{chunk.short_name}.parquet"
            )

        ParallelPbar("Joining full with agg")(n_jobs=n_jobs)(
            delayed(process_chunk)(chunk) for chunk in chunks
        )

    @staticmethod
    def join_in_chunks(
            export_folder: Path,
            # Left join so the first one needs to be full
            lfs_to_join: List[LazyFrame],
            join_on: List[str] = ["i", "j", "face"],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=3),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        if lfs_to_join[0].filter(
                pl.col("i") == 1).collect().height != 8192 * 6:
            print("Cannot do merge as first input lf is not full for i, j and face")
            return

        def process_chunk(chunk) -> None:
            filtered: List[pl.LazyFrame] = [
                chunk.filter_lf(df)
                for df in lfs_to_join
            ]

            combined: LazyFrame = reduce(
                lambda left, right: left.join(
                    right,
                    on=join_on,
                    how="left",
                    coalesce=True,
                ),
                filtered,
            )

            combined.sink_parquet(
                export_folder / f"{chunk.short_name}.parquet")

        for chunk in tqdm(chunks, desc="Joining"):
            process_chunk(chunk)

    @staticmethod
    def agg_in_slices(
            export_df_path: Path,
            lf_to_agg: LazyFrame,
            agg_group: str,
            agg_exprs: List[Expr],
            slice_size: int = 1_000,
            skip_if_exists=False, n_jobs=4) -> None:

        if export_df_path.exists() and skip_if_exists:
            return

        export_df_path.parent.mkdir(exist_ok=True, parents=True)
        groups: pl.Series = lf_to_agg.group_by(
            agg_group).agg().collect()[agg_group].sort()
        print(f"Found {len(groups)} groups")

        group_slices: List[pl.Series] = [
            groups.slice(i, slice_size)
            for i in range(0, len(groups), slice_size)
        ]

        def process_group_slice(group_slice: pl.Series) -> DataFrame:
            agg_data: DataFrame = lf_to_agg.filter(pl.col(agg_group).is_in(
                group_slice.implode())).collect().group_by(
                    agg_group
            ).agg(*agg_exprs)

            return agg_data

        agg_data_dfs: List[DataFrame | None] = list(ParallelPbar("Joining")(n_jobs=n_jobs)(
            delayed(process_group_slice)(group_slice) for group_slice in group_slices
        ))

        merged_df: pl.DataFrame = pl.concat(agg_data_dfs)
        merged_df.write_parquet(export_df_path)

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
