import os
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from pds4_tools import pds4_read
from pds4_tools.reader.general_objects import StructureList

from boulder_statistics.analysis.external_data_encyclopedia import \
    ExternalDataEncyclopedia


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
                })

                pds4_dfs.append(pds4_df)

                if verbose:
                    print(f"{file_name} done")

        merged_pds4_df: pl.DataFrame = pl.concat(pds4_dfs, how="diagonal")

        if cache_file_path is not None:
            merged_pds4_df.write_parquet(cache_file_path)

        return merged_pds4_df
