import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import polars as pl
from lxml import etree
from pds4_tools import pds4_read
from pds4_tools.reader.general_objects import StructureList

TIR_COLUMN_NAMES: List[str] = ["Slope1000", "BD350", "BD440", "Ratio1000"]
VNIR_COLUMN_NAMES: List[str] = [
    "BandArea3200to3600nm",
    "OH2700nm",
    "Pyroxene920nm",
    "Refl550nm",
    "Slope1polyfit",
    "Slope2polyfit"]

MEASUREMENT_FILE_TO_COLUMN_LOOKUP: dict[str, str] = {
    # TIR
    "_slope1000_": "Slope1000",
    "_bd350_": "BD350",
    "_bd440_": "BD440",
    "_ratio1000_": "Ratio1000",

    # VNIR
    "_bandarea3200to3600nm_": "BandArea3200to3600nm",
    "_oh27": "OH2700nm",
    "_pyroxene920nm": "Pyroxene920nm",
    "_refl550nm_": "Refl550nm",
    "_slope1poly_": "Slope1polyfit",
    "_slope2": "Slope2polyfit",
}


@dataclass
class BulkParseDataMaps():
    data_type: Literal["TIR", "VNIR"]
    cache_file_path: Path | None
    data_maps_path: Path  # From ExternalDataEncyclopedia in most cases

    @property
    def measurement_names(self) -> List[str]:
        match self.data_type:
            case "TIR":
                return TIR_COLUMN_NAMES
            case "VNIR":
                return VNIR_COLUMN_NAMES

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

    def get_table_name(self, xml_path: Path) -> str:
        tree = etree.parse(xml_path)

        namespaces = {
            "pds": "http://pds.nasa.gov/pds4/pds/v1"
        }

        name = tree.find(
            ".//pds:Table_Binary/pds:name",
            namespaces
        )

        if name is None:
            raise ValueError(f"No Table_Binary name found in {xml_path}")

        return name.text

    def find_measurement_name(self, columns_to_extract: List[str]) -> str:
        excluded: set[str] = {
            "SIGMA",
            "RADIUS",
            "LONGITUDE",
            "LATITUDE",
            "FACET_NUM"}

        remaining: List[str] = [
            name for name in columns_to_extract if name not in excluded]

        assert len(remaining) == 1, (
            f"Expected exactly one non-standard column, found {remaining}"
        )

        column_name = remaining[0]

        return column_name

    def column_name_from_XML(self, xml_path: Path) -> str:
        matches: List[str] = [
            value
            for key, value in MEASUREMENT_FILE_TO_COLUMN_LOOKUP.items()
            if key in xml_path.stem
        ]

        assert len(matches) == 1, (
            f"Expected exactly one measurement column for '{xml_path.stem}', "
            f"found {matches}"
        )

        return matches[0]

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

                pds4_df = (pl.DataFrame({
                    column.lower(): struc_data[column].astype(np.float64) for column in column_names_to_extract
                })
                    .rename({  # To avoid duplications we set the names manually
                        self.find_measurement_name(column_names_to_extract).lower(): self.column_name_from_XML(pds4_xml_path)
                    })
                    .with_columns(
                    pl.lit(mission_phase_folder_name).alias(
                        "mission_phase"
                    ),
                    pl.lit(facet_shape_model_name).alias(
                        "facet_shape_model_name"),
                )
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
