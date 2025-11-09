import igl
import polars as pl
from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


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

        verts, faces = igl.read_triangle_mesh(path.actual_path.as_posix())

        points: pl.DataFrame = pl.DataFrame(
            verts, schema=[
                "x", "y", "z"]).with_row_index("vid")
        tris = pl.DataFrame(faces, schema=["0", "1", "2"])

        return (points, tris)

    def write(self, obj: tuple[pl.DataFrame, pl.DataFrame],
              path: FSPathLocalDisk) -> None:
        raise NotImplementedError()
