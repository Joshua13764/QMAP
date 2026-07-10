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
    "reconc": "g_01600mm_spc_obj_0000n00000_v042.obj"
}

TIR_MEASUREMENT_NAMES: List[str] = [
    "band depth 350",
    "band depth 440",
    "slope 1000",
    "ratio 1000"]

TIR_SIGMA_MEASUREMENT_NAMES: List[str] = [
    f"sigma {tir_measurement_name}"
    for tir_measurement_name in TIR_MEASUREMENT_NAMES
]


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
                    pl.lit(mission_phase_folder_name).alias(
                        "mission_phase"
                    ),
                    pl.lit(facet_shape_model_name).alias(
                        "facet_shape_model_name")
                )

                pds4_dfs.append(pds4_df)

                if verbose:
                    print(f"{file_name} done")

        merged_pds4_df: pl.DataFrame = pl.concat(pds4_dfs, how="diagonal").with_columns(
            x_hat=pl.col("latitude").radians().cos() *
            pl.col("longitude").radians().cos(),

            y_hat=pl.col("latitude").radians().cos() *
            pl.col("longitude").radians().sin(),

            z_hat=pl.col("latitude").radians().sin(),

        ).with_columns(
            x=pl.col("x_hat") * pl.col("radius"),
            y=pl.col("y_hat") * pl.col("radius"),
            z=pl.col("z_hat") * pl.col("radius")

        ).group_by("facet_num", "mission_phase").agg(
            *[
                pl.col(tir_measurement_name)
                .filter(pl.col(tir_measurement_name).is_not_null())
                .first()
                .alias(tir_measurement_name)
                for tir_measurement_name in TIR_MEASUREMENT_NAMES
            ],
            *[
                pl.col("sigma")
                .filter(pl.col(tir_measurement_name).is_not_null())
                .first()
                .alias(tir_sigma_measurement_name)
                for tir_measurement_name, tir_sigma_measurement_name in zip(
                    TIR_MEASUREMENT_NAMES, TIR_SIGMA_MEASUREMENT_NAMES)
            ],
            pl.col("x").first().alias("x"),
            pl.col("y").first().alias("y"),
            pl.col("z").first().alias("z"),
            pl.len().alias("count")
        ).with_columns(
            pl.col("facet_num").cast(pl.Int32)
        )

        if cache_file_path is not None:
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            merged_pds4_df.write_parquet(cache_file_path)

        return merged_pds4_df
