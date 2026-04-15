from dataclasses import dataclass

import igl
import polars as pl

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData
from boulder_statistics.steps.utils.polars_3D_expressions import (POINT_ATTRS,
                                                                  VERT_ID_COLS)


@dataclass(frozen=True)
class FSPolarsObjAdapterFastPLOBJDataSpecifyPath(
        FSAdapterBase[PLOBJData, FSPathLocalDisk]):

    mesh_path: str

    def read(self, path: FSPathLocalDisk) -> PLOBJData:
        """_summary_

        Args:
            path (FSPathLocalDisk): _description_

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: (points, tris)

            points : with headers "x", "y", "z" and "vid" (row id 0-index)
            tris : with headers "0", "1", "2"
        """

        verts, faces = igl.read_triangle_mesh(self.mesh_path)

        return PLOBJData(
            verts=pl.LazyFrame(
                verts, schema=POINT_ATTRS).with_row_index("vid"),
            tris=pl.LazyFrame(faces, schema=VERT_ID_COLS)
        )

    def write(self, obj: PLOBJData,
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
