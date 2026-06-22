from typing import Dict

import numpy as np
import polars as pl
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from tqdm import tqdm


def get_longest_axis_diameter_lookup(
        mask_atlas_combined: pl.LazyFrame) -> Dict[np.uint32, np.float32]:

    filtered: pl.LazyFrame = (
        mask_atlas_combined
        .filter(
            (pl.col("positions_x").abs() < 100) &
            (pl.col("positions_y").abs() < 100) &
            (pl.col("positions_z").abs() < 100)
        )
        .group_by("row_id")
        .agg([
            pl.col("positions_x"),
            pl.col("positions_y"),
            pl.col("positions_z"),
        ])
    )

    df: pl.DataFrame = filtered.collect(engine="streaming")

    longest_axis_diameters: Dict[np.uint32, np.float32] = {}
    surface_areas: Dict[np.uint32, np.float32] = {}

    for row in tqdm(df.iter_rows(named=True),
                    "Processing longest_axis_diameters", total=df.height):
        positions = np.vstack([
            row["positions_x"],
            row["positions_y"],
            row["positions_z"]
        ]).T

        if len(positions) < 4:
            longest_axis_diameters[row["row_id"]] = np.float32(np.nan)
            continue

        hull = ConvexHull(positions)
        hull_points = positions[hull.vertices]

        longest_axis = np.max(pdist(hull_points)).astype(np.float32)
        surface_areas[row["row_id"]] = np.float32(hull.area)
        longest_axis_diameters[row["row_id"]] = longest_axis

    return longest_axis_diameters, surface_areas
