from dataclasses import dataclass

import igl
import polars as pl

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from Boulder_Statistics.steps.utils.polars_3D_expressions import (POINT_ATTRS,
                                                                  VERT_ID_COLS)


@dataclass(frozen=True)
class FSPolarsObjAdapterFast(
        FSAdapterBase[tuple[pl.LazyFrame, pl.LazyFrame], FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """_summary_

        Args:
            path (FSPathLocalDisk): _description_

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: (points, tris)

            points : with headers "x", "y", "z" and "vid" (row id 0-index)
            tris : with headers "0", "1", "2"
        """

        verts, faces = igl.read_triangle_mesh(path.actual_path.as_posix())

        points: pl.LazyFrame = pl.LazyFrame(
            verts, schema=POINT_ATTRS).with_row_index("vid")
        tris = pl.LazyFrame(faces, schema=VERT_ID_COLS)

        return (points, tris)

    def write(self, obj: tuple[pl.LazyFrame, pl.LazyFrame],
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
