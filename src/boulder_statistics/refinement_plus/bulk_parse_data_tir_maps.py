import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
from pds4_tools import pds4_read
from pds4_tools.reader.general_objects import StructureList

from boulder_statistics.analysis.external_data_encyclopedia import \
    ExternalDataEncyclopedia

FACET_SHAPE_MODELS: Dict[str, str] = {
    "detailed_survey": "g_06310mm_spc_obj_0000n00000_v020.obj",
    "recona": "g_01600mm_spc_obj_0000n00000_v042.obj",
    "reconb": "g_01600mm_spc_obj_0000n00000_v042.obj",
    "reconc": "g_00800mm_spc_obj_0000n00000_v042.obj"
}


class DataTirMaps:
    @staticmethod
    def bulk_parse(ed: ExternalDataEncyclopedia,
                   cache_file_path: Path | None = Path(
                       ".cache/data_tir_maps_parse_cache.parquet"),
                   verbose=False) -> pl.DataFrame:

        if (cache_file_path is not None) and cache_file_path.exists():
            return pl.read_parquet(cache_file_path)

        pds4_dfs: List[pl.DataFrame] = []

        for mission_phase_folder_name in os.listdir(ed.data_tir_maps_path):
            mission_phase_folder_path: Path = ed.data_tir_maps_path / mission_phase_folder_name
            facet_shape_model_name = FACET_SHAPE_MODELS[mission_phase_folder_name]

            for file_name in os.listdir(mission_phase_folder_path):
                if ".xml" not in file_name:
                    continue

                pds4_xml_path: Path = mission_phase_folder_path / file_name
                struc: StructureList = pds4_read(
                    pds4_xml_path.as_posix(), quiet=True, lazy_load=True)
                struc_data = struc[2].data

                column_names_to_extract = struc_data.dtype.names

                pds4_df = pl.DataFrame({
                    column.lower(): struc_data[column].astype(np.float64) for column in column_names_to_extract
                }).with_columns(
                    pl.lit(facet_shape_model_name).alias(
                        "facet_shape_model_name")
                )

                pds4_dfs.append(pds4_df)

                if verbose:
                    print(f"{file_name} done")

        merged_pds4_df: pl.DataFrame = pl.concat(pds4_dfs, how="diagonal")

        if cache_file_path is not None:
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            merged_pds4_df.write_parquet(cache_file_path)

        return merged_pds4_df.with_columns(
            x_hat=pl.col("latitude").radians().cos() *
            pl.col("longitude").radians().cos(),
            y_hat=pl.col("latitude").radians().cos() *
            pl.col("longitude").radians().sin(),
            z_hat=pl.col("latitude").radians().sin(),

        ).with_columns(
            x=pl.col("x_hat") * pl.col("radius"),
            y=pl.col("y_hat") * pl.col("radius"),
            z=pl.col("z_hat") * pl.col("radius")

        ).group_by("facet_num", "facet_shape_model_name").agg(
            pl.col("band depth 350")
            .filter(pl.col("band depth 350").is_not_null())
            .first()
            .alias("band depth 350"),

            pl.col("band depth 440")
            .filter(pl.col("band depth 440").is_not_null())
            .first()
            .alias("band depth 440"),

            pl.col("slope 1000")
            .filter(pl.col("slope 1000").is_not_null())
            .first()
            .alias("slope 1000"),

            pl.col("ratio 1000")
            .filter(pl.col("ratio 1000").is_not_null())
            .first()
            .alias("ratio 1000"),

            pl.col("sigma")
            .filter(pl.col("band depth 350").is_not_null())
            .first()
            .alias("sigma band depth 350"),

            pl.col("sigma")
            .filter(pl.col("band depth 440").is_not_null())
            .first()
            .alias("sigma band depth 440"),

            pl.col("sigma")
            .filter(pl.col("slope 1000").is_not_null())
            .first()
            .alias("sigma slope 1000"),

            pl.col("sigma")
            .filter(pl.col("ratio 1000").is_not_null())
            .first()
            .alias("sigma ratio 1000"),

            pl.col("x").first(),
            pl.col("y").first(),
            pl.col("z").first(),
            pl.len().alias("count")
        )
