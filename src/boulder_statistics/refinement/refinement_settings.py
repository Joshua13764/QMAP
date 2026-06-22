from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame


@dataclass(kw_only=True, frozen=True)
class RefinementSettings():
    data_products_path: Path
    figures_path: Path
    raw_database_file_path: Path
    refinement_cache_path: Path

    merge_threshold: float = field(default=0.05)
    tile_offset_lookup: dict[str, np.typing.NDArray[np.uint32]] = field(default_factory=lambda: {  # Like a matrix index (ij)
        "A": np.array([0, 0], dtype=np.uint32),
        "B": np.array([0, 512], dtype=np.uint32),
        "C": np.array([512, 0], dtype=np.uint32),
        "D": np.array([512, 512], dtype=np.uint32)
    })

    calculate_tile_pixel_offset: Callable[[str], np.typing.NDArray[np.uint32]] = field(default=lambda tile_lod_code: sum([(2 ** i) * tile_offset_lookup[t] for i, t in enumerate(
        tile_lod_code[::-1])]) + np.zeros(2, dtype=np.uint32))

    @property
    def cache_path_part_0(
        self) -> Path: return self.refinement_cache_path / "refinement_part_0"

    @property
    def cache_path_part_1(
        self) -> Path: return self.refinement_cache_path / "refinement_part_1"

    @property
    def cache_path_part_3_3(
        self) -> Path: return self.refinement_cache_path / "refinement_part_3_3"

    @property
    def cache_path_part_3_4(
        self) -> Path: return self.refinement_cache_path / "refinement_part_3_4"

    @property
    def combined_atlas_path(self) -> Path:
        return self.data_products_path / "combined_atlas.parquet"

    @property
    def combined_mask_path(self) -> Path:
        return self.data_products_path / "combined_mask.parquet"

    @property
    def combined_mask_no_merge_path(self) -> Path:
        return self.data_products_path / "combined_mask_no_merge.parquet"

    @property
    def mask_atlas_combined_path(self) -> Path:
        return self.data_products_path / "mask_atlas_combined.parquet"

    @property
    def boulder_agg_data_path(self) -> Path:
        return self.data_products_path / "boulder_agg_data.parquet"

    @property
    def combined_mask(
        self) -> LazyFrame: return pl.scan_parquet(self.combined_mask_path)

    @property
    def combined_atlas(
        self) -> LazyFrame: return pl.scan_parquet(self.combined_atlas_path)

    @property
    def mask_atlas_combined(
        self) -> LazyFrame: return pl.scan_parquet(self.mask_atlas_combined_path)

    @property
    def Phi_export_path_mesh(
        self) -> Path: return self.cache_path_part_3_3 / "Phi_export_mesh"

    @property
    def Phi_export_path_sphere(
        self) -> Path: return self.cache_path_part_3_3 / "Phi_export_sphere"
