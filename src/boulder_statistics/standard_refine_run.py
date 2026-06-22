from pathlib import Path

import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame

from boulder_statistics.refinement.database_metadata import DatabaseMetadata
from boulder_statistics.refinement.refinement_settings import \
    RefinementSettings
from boulder_statistics.refinement.utils.step_0_0 import create_merge_db_cache
from boulder_statistics.refinement.utils.step_0_1 import \
    export_combined_mask_and_merge
from boulder_statistics.refinement.utils.step_1 import (
    create_combined_atlas_components, merge_combined_atlas_components)
from boulder_statistics.refinement.utils.step_2 import \
    join_mask_and_atlas_tables
from boulder_statistics.refinement.utils.step_3_1 import \
    get_longest_axis_diameter_lookup
from boulder_statistics.refinement.utils.step_3_2 import agg_stats
from boulder_statistics.refinement.utils.step_3_3 import find_LAS
from boulder_statistics.refinement.utils.step_3_4 import run_compute_Phi_counts


class StandardRefineRun():
    @staticmethod
    # --- Step 0 : Merge detections ---
    def step_0(
            rs: RefinementSettings) -> tuple[DataFrame, DatabaseMetadata]:
        lf: LazyFrame = pl.scan_csv(rs.raw_database_file_path)

        # Add some additional metadata
        # Very important that row ids are assigned in blacks for tile lod_code
        # for create_merge_db_cache
        lf = lf.sort("tile_lod_code")
        lf = lf.with_row_index("row_id")
        df = lf.collect()

        df_meta: DatabaseMetadata = DatabaseMetadata.from_df(df)

        cache_folder: Path = rs.cache_path_part_0
        merge_db_cache: Path = cache_folder / "merge_db_cache"
        merge_db_cache.mkdir(parents=True, exist_ok=True)

        if not rs.combined_mask_no_merge_path.exists():

            active_row_ids: pl.DataFrame = create_merge_db_cache(
                df=df,
                faces=df_meta.faces,
                lod_levels=df_meta.lods,
                cache_folder=merge_db_cache,
                calculate_tile_pixel_offset=rs.calculate_tile_pixel_offset,
                combined_mask_no_merge_path=rs.combined_mask_no_merge_path
            )

        print("Finding active row ids...")
        active_row_ids: DataFrame = pl.scan_parquet(
            rs.combined_mask_no_merge_path).group_by("row_id").agg().collect()

        df: DataFrame = df.join(active_row_ids, on="row_id", how="inner").select(
            "row_id", "tile_face", "tile_lod_number", "tile_lod_code", "tile_reciprocal_length", "tile_reciprocal_area",
            "tile_x_min", "tile_x_max", "tile_y_min", "tile_y_max"
        )

        merged_df_path: Path = cache_folder / "merged_df.parquet"

        if any([not rs.combined_mask_path.exists(),
               not merged_df_path.exists()]):

            export_combined_mask_and_merge(
                df=df,
                faces=df_meta.faces,
                cache_folder=cache_folder,
                combined_mask_path=rs.combined_mask_path,
                merge_threshold=rs.merge_threshold,
                combined_mask_no_merge_path=rs.combined_mask_no_merge_path,
                merged_df_path=merged_df_path
            )

        df: DataFrame = pl.read_parquet(merged_df_path)

        return df, df_meta

    @staticmethod
    # --- Step 1 : Create attribute atlas ---
    def step_1(rs: RefinementSettings,
               df: DataFrame, df_meta: DatabaseMetadata) -> None:
        create_combined_atlas_components(
            df=df,
            faces=df_meta.faces,
            max_lod_level=df_meta.max_lod,
            calculate_tile_pixel_offset=rs.calculate_tile_pixel_offset,
            attribute_atlas_cache_folder=rs.cache_path_part_1
        )

        if not rs.combined_atlas_path.exists():

            merge_combined_atlas_components(
                combined_atlas_path=rs.combined_atlas_path
            )

    @staticmethod
    # --- Step 2 : Fill combined merge with the relevant atlas data ---
    def step_2(rs: RefinementSettings, df_meta: DatabaseMetadata) -> None:

        if not rs.mask_atlas_combined_path.exists():

            join_mask_and_atlas_tables(
                faces=df_meta.faces,
                combined_mask_path=rs.combined_mask_path,
                combined_atlas_path=rs.combined_atlas_path,
                mask_atlas_combined_path=rs.mask_atlas_combined_path
            )

    @staticmethod
    # --- Step 3.1 : Finding longest axis for each boulder in meters ---
    def step_3_1(rs: RefinementSettings, df: DataFrame) -> DataFrame:

        df = pl.read_parquet(rs.boulder_agg_data_path) if rs.boulder_agg_data_path.exists() else \
            df.select(["row_id", "tile_face",
                      "tile_lod_number", "tile_lod_code"])

        if "longest_axis_diameter" not in df.columns:

        longest_axis_diameters, surface_areas = get_longest_axis_diameter_lookup(
            rs.mask_atlas_combined)

        df = df.with_columns(
            pl.col("row_id")
            .replace_strict(longest_axis_diameters, default=None)
            .cast(pl.Float32)
            .alias("longest_axis_diameter"),

            pl.col("row_id")
            .replace_strict(surface_areas, default=None)
            .cast(pl.Float32)
            .alias("surface_area")
        )

        df.write_parquet(rs.boulder_agg_data_path)

        return df

    @staticmethod
    # --- Step 3.2 : Aggregate data ---
    def step_3_2(rs: RefinementSettings, df: DataFrame) -> DataFrame:

        if "mean_i" not in df.columns:

            agg_df: DataFrame = agg_stats(rs.mask_atlas_combined)

            df = df.join(
                agg_df,
                on=["row_id"],
                how="inner"
            )

            df.write_parquet(rs.boulder_agg_data_path)

        return df

    @staticmethod
    # --- Step 3.3 : Find LAS ---
    def step_3_3(rs: RefinementSettings, df_meta: DatabaseMetadata) -> None:
        cache_folder: Path = rs.cache_path_part_3_3
        cache_folder.mkdir(exist_ok=True)

        find_LAS(
            cache_folder=cache_folder,
            faces=df_meta.faces,
            num_i=df_meta.num_i,
            combined_atlas=rs.combined_atlas,
            Phi_export_path_mesh=rs.Phi_export_path_mesh,
            Phi_export_path_sphere=rs.Phi_export_path_sphere
        )

    @staticmethod
    # --- step 3.4 : Find Phi Counts ---
    def step_3_4(rs: RefinementSettings) -> None:
        phi_counts_cache_folder = Path("refinement_part_3_4")

        Phi_counts_smoothed_path: Path = phi_counts_cache_folder / "Phi_counts_smoothed"
        Phi_counts_noisy_path: Path = phi_counts_cache_folder / "Phi_counts_noisy"
        Phi_counts_sphere_theoretical_path: Path = phi_counts_cache_folder / \
            "Phi_counts_sphere_theoretical"
        Phi_counts_sphere_noisy_path: Path = phi_counts_cache_folder / "Phi_counts_sphere_noisy"

        run_compute_Phi_counts(
            Phi_counts_smoothed_path, Phi_counts_noisy_path,
            Phi_counts_sphere_theoretical_path, Phi_counts_sphere_noisy_path,

            Phi_mesh=pl.scan_parquet(rs.Phi_export_path_mesh),
            Phi_sphere=pl.scan_parquet(rs.Phi_export_path_sphere),
            bin_numbers=np.array([128, 256, 512, 1024, 2048, 4096])
        )

    @staticmethod
    def run_standard_refinement(
            refinement_settings: RefinementSettings) -> None:
        # --- Step -1 : make required folders ---
        refinement_settings.data_products_path.mkdir(
            parents=True, exist_ok=True)
        refinement_settings.figures_path.mkdir(parents=True, exist_ok=True)

        df, df_meta = StandardRefineRun.step_0(refinement_settings)
        StandardRefineRun.step_1(refinement_settings, df, df_meta)
        StandardRefineRun.step_2(refinement_settings, df_meta)
        df = StandardRefineRun.step_3_1(refinement_settings, df)
        df = StandardRefineRun.step_3_2(refinement_settings, df)
        StandardRefineRun.step_3_3(refinement_settings, df_meta)
        StandardRefineRun.step_3_4(refinement_settings)
