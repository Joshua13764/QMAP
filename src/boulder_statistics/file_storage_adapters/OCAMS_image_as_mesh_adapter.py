from dataclasses import dataclass

import numpy as np
import polars as pl
import tifffile

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData
from boulder_statistics.steps.utils.polars_3D_expressions import (POINT_ATTRS,
                                                                  VERT_ID_COLS)


@dataclass(frozen=True)
class FSOCAMSImageAsMeshAdapter(
        FSAdapterBase[PLOBJData, FSPathLocalDisk]):
    positions_path: str
    colors_path: str

    def read(self, path: FSPathLocalDisk) -> PLOBJData:
        positions = tifffile.imread(self.positions_path)
        colors = tifffile.imread(self.colors_path)

        flattened_indices = np.arange(
            colors.size).reshape(
            colors.shape[0],
            colors.shape[1])

        v00 = flattened_indices[:-1, :-1]
        v10 = flattened_indices[1:, :-1]
        v01 = flattened_indices[:-1, 1:]
        v11 = flattened_indices[1:, 1:]

        top_left = np.stack([v00, v10, v01], axis=-1)
        bottom_right = np.stack([v10, v11, v01], axis=-1)

        triangles = np.concatenate([
            top_left.reshape(-1, 3),
            bottom_right.reshape(-1, 3)
        ], axis=0)

        positions_flat = positions.reshape(-1, 3)
        colors_flat = colors.flatten()

        verts_df = pl.DataFrame(
            positions_flat,
            schema=["x", "y", "z"]
        ).with_row_index("vid")

        tris_df = pl.DataFrame(
            triangles,
            schema=["0", "1", "2"]
        ).with_columns(pl.Series("color", np.mean(colors_flat[triangles], axis=1)))

        incorrect_verts = verts_df.filter(
            (pl.col("x") == pl.lit(np.nan)) |
            (pl.col("y") == pl.lit(np.nan)) |
            (pl.col("z") == pl.lit(np.nan)))["vid"].to_numpy()

        # verts_filtered = verts_df.filter(
        #     (pl.col("x") != pl.lit(np.nan)) &
        #     (pl.col("y") != pl.lit(np.nan)) &
        #     (pl.col("z") != pl.lit(np.nan)))

        tris_filtered = tris_df.filter(
            ~pl.any_horizontal(pl.all().is_in(incorrect_verts))
        )

        return PLOBJData(
            verts=verts_df.lazy(),
            tris=tris_filtered.lazy()
        )

    def write(self, obj: PLOBJData,
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
