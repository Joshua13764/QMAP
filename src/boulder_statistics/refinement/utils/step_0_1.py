from polars import DataFrame, LazyFrame
import polars as pl
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from pathlib import Path
import graphviz
import re

def identify_rows_to_remove(faces : List[str], merge_db_cache : LazyFrame, merge_threshold : float) -> List[int]:
    rows_to_remove: list[int] = list()

    for face in faces:

        combined_merge_data_lf: pl.LazyFrame = merge_db_cache.filter(pl.col("face") == face)

        row_counts: pl.LazyFrame = (
            combined_merge_data_lf
            .group_by("row_id")
            .agg(pl.len().alias("pixel_count"), pl.col("lod_code").first(), pl.col("lod_level").first())
        )

        pair_counts: pl.LazyFrame = (
            combined_merge_data_lf
            .join(
                combined_merge_data_lf,
                on=["i", "j"],
                how="inner",
                suffix="_overlap"
            )
            .group_by(["row_id", "row_id_overlap"])
            .agg(pl.len().alias("shared_pixels"))
        )

        collected_pairs: pl.LazyFrame = (
            pair_counts
            .join(
                row_counts,
                on="row_id",
                how="left"
            )
            .join(
                row_counts.rename({
                    "row_id": "row_id_overlap",
                    "pixel_count": "pixel_count_overlap",
                    "lod_code": "lod_code_overlap",
                    "lod_level" : "lod_level_overlap"
                }),
                on="row_id_overlap",
                how="left"
            )
        )

        for row_id, group in collected_pairs.collect().partition_by("row_id", as_dict=True).items():

            merge_df: DataFrame = group.with_columns(
                (pl.col("shared_pixels") / (pl.col("pixel_count") + pl.col("pixel_count_overlap") - pl.col("shared_pixels"))).alias("jaccard_index"),
                (pl.col("shared_pixels") / pl.col("pixel_count")).alias("proportion_overlap"),
                (pl.col("pixel_count_overlap") / pl.col("pixel_count")).alias("overlap_scale_factor")
            )

            number_of_larger_good_merges = merge_df.filter(
                (pl.col("jaccard_index") > pl.lit(merge_threshold)) & (pl.col("overlap_scale_factor") >= 1)).height

            if number_of_larger_good_merges > 1:
                rows_to_remove.append(row_id[0])

        print(f"Number of rows removed so far are {len(rows_to_remove)}")

    # Logical plan @ 300 DPI
    render_high_res(collected_pairs, "logical_plan", optimized=False, dpi=300)

    # Optimized plan @ 300 DPI
    render_high_res(collected_pairs, "optimized_plan", optimized=True, dpi=300)

    return rows_to_remove

def render_high_res(lf, name, optimized, figures_path : Path = Path("figures"), dpi=300) -> None:
    # Get raw DOT from Polars
    dot = lf.show_graph(optimized=optimized, raw_output=True)

    # If there’s already a graph header, insert DPI
    # e.g., "digraph polars_query_plan {"
    dot = re.sub(
        r"^(digraph [^{]+{)",
        r"\1\ngraph [dpi=%d];" % dpi,
        dot,
        flags=re.MULTILINE
    )

    graph = graphviz.Source(dot)

    graph.render(
        filename=name,
        directory=figures_path,
        format="png",
        cleanup=True
    )

def export_combined_mask_and_merge(df : DataFrame, faces : List[str], cache_folder : Path, combined_mask_path : Path, combined_mask_no_merge_path : Path, merge_threshold : float,
                                   merged_df_path : Path, merge = True) -> None:
    
    combined_mask_no_merge = pl.scan_parquet(combined_mask_no_merge_path)

    rows_to_remove: List[int] = identify_rows_to_remove(
        faces, combined_mask_no_merge, merge_threshold) if merge else []
    
    # Export post merge as a data_product
    combined_mask_no_merge.filter(
            ~pl.col("row_id").is_in(rows_to_remove)
        ).sink_parquet(combined_mask_path)

    df.filter(~pl.col("row_id").is_in(rows_to_remove)).write_parquet(merged_df_path)

