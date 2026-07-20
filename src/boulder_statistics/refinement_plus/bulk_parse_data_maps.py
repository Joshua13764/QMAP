import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from lxml import etree
from pds4_tools import pds4_read
from pds4_tools.reader.general_objects import StructureList


@dataclass
class BulkParseDataMaps():
    measurement_names: List[str]
    cache_file_path: Path | None
    data_maps_path: Path  # From ExternalDataEncyclopedia in most cases

    @property
    def get_all_measurement_types_of_interest(self) -> List[str]:
        return self.measurement_names + self.sigma_measurement_names

    @property
    def sigma_measurement_names(self) -> List[str]:
        return [
            f"sigma {measurement_name}"
            for measurement_name in self.measurement_names
        ]

    def get_shape_model_name(self, xml_path: Path) -> str:
        tree = etree.parse(xml_path)

        namespaces = {
            "orex": "http://pds.nasa.gov/pds4/mission/orex/v1"
        }

        obj_file = tree.find(
            ".//orex:Shape_Data_Source/orex:obj_file",
            namespaces
        )

        if obj_file is None:
            raise ValueError(f"No obj_file found in {xml_path}")

        return obj_file.text

    def bulk_parse(self, verbose=False) -> pl.DataFrame:

        if (self.cache_file_path is not None) and self.cache_file_path.exists():
            return pl.read_parquet(self.cache_file_path)

        pds4_dfs: List[pl.DataFrame] = []

        for mission_phase_folder_name in os.listdir(self.data_maps_path):
            mission_phase_folder_path: Path = self.data_maps_path / mission_phase_folder_name

            for file_name in os.listdir(mission_phase_folder_path):
                if ".xml" not in file_name:
                    continue

                pds4_xml_path: Path = mission_phase_folder_path / file_name

                facet_shape_model_name: str = self.get_shape_model_name(
                    pds4_xml_path)

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
                pl.col(measurement_name)
                .filter(pl.col(measurement_name).is_not_null())
                .first()
                .alias(measurement_name)
                for measurement_name in self.measurement_names
            ],

            *[
                pl.col("sigma")
                .filter(pl.col(measurement_name).is_not_null())
                .first()
                .alias(sigma_measurement_name)
                for measurement_name, sigma_measurement_name in zip(
                    self.measurement_names, self.sigma_measurement_names)
            ],

            pl.col("x").first().alias("x"),
            pl.col("y").first().alias("y"),
            pl.col("z").first().alias("z"),

            pl.col("facet_shape_model_name").first(),

            pl.len().alias("count")
        ).with_columns(
            pl.col("facet_num").cast(pl.Int32)
        )

        if self.cache_file_path is not None:
            self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            merged_pds4_df.write_parquet(self.cache_file_path)

        return merged_pds4_df
