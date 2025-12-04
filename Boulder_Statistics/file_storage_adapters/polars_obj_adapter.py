from dataclasses import dataclass
from typing import List

import polars as pl
import trimesh

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from Boulder_Statistics.steps.utils.polars_3D_expressions import (POINT_ATTRS,
                                                                  VERT_ID_COLS)

IDX_COL_HEADERS: List[str] = ["0", "1", "2"]


@dataclass(frozen=True)
class FSPolarsObjAdapter(
        FSAdapterBase[tuple[pl.DataFrame, pl.DataFrame], FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> tuple[pl.DataFrame, pl.DataFrame]:
        """_summary_

        Args:
            path (FSPathLocalDisk): _description_

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: (points, tris)

            points : with headers "x", "y", "z" and "vid" (row id 0-index)
            tris : with headers "0", "1", "2"
        """
        mesh: trimesh.Trimesh = trimesh.load_mesh(
            path.actual_path.as_posix(), file_type="obj", process=True)

        points: pl.DataFrame = pl.DataFrame(
            mesh.vertices.tolist(), schema=POINT_ATTRS, orient="row").with_row_index("vid")
        tris: pl.DataFrame = pl.DataFrame(
            mesh.faces.tolist(), schema=VERT_ID_COLS, orient="row")

        return (points, tris)

    def write(self, obj: tuple[pl.DataFrame, pl.DataFrame],
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
