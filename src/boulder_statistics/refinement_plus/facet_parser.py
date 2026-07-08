from pathlib import Path
from typing import Callable, Tuple

import datashader as ds
import datashader.utils as du
import igl
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree

from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData
from boulder_statistics.refinement_plus.qcube_chunk import QCubeChunk
from boulder_statistics.steps.utils.Better_polars_3D_expressions import (
    POINT_ATTRS, VERT_ID_COLS, BetterPolars3DExpressions)


class FacetParser:

    @staticmethod
    def load_mesh(mesh_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
        # Meshes can be found here
        # https://sbnarchive.psi.edu/pds4/orex/orex.altimetry/data_derived_altimetry_global_models/global_digital_terrain_models/SPCv20/

        verts, faces = igl.read_triangle_mesh(mesh_path)

        obj_data = PLOBJData(
            verts=pl.LazyFrame(
                verts, schema=POINT_ATTRS).with_row_index("vid"),
            tris=pl.LazyFrame(faces, schema=VERT_ID_COLS)
        )

        points: pl.LazyFrame = BetterPolars3DExpressions._project_points(
            obj_data.verts)
        tris: pl.LazyFrame = BetterPolars3DExpressions._attach_points_to_tris(
            points, obj_data.tris)
        tris: pl.LazyFrame = tris.with_columns(
            BetterPolars3DExpressions.get_mean_radius(),
            BetterPolars3DExpressions.get_mean_x().alias("x_tri_mean"),
            BetterPolars3DExpressions.get_mean_y().alias("y_tri_mean"),
            BetterPolars3DExpressions.get_mean_z().alias("z_tri_mean")
        )

        return points.collect(), tris.collect()

    @staticmethod
    def associate_mesh_tris_with_facet_num(
            facets: pl.DataFrame, mesh_tris: pl.DataFrame):
        facets_xyz = (
            facets
            .select(["x", "y", "z"])
            .to_numpy()
            .astype(np.float32)
        )

        tris_xyz = (
            mesh_tris
            .select(["x_tri_mean", "y_tri_mean", "z_tri_mean"])
            .to_numpy()
            .astype(np.float32)
        )

        chunk_tree = cKDTree(facets_xyz)

        distance, idx = chunk_tree.query(
            tris_xyz,
            k=1,
            workers=-1,
        )

        facet_nums = facets["facet_num"].to_numpy()

        # valid nearest neighbours
        valid = idx < len(facet_nums)

        nearest_facet_num = np.full(len(idx), np.nan)
        nearest_facet_num = facet_nums[idx[valid]]

        return mesh_tris.with_columns(
            pl.Series("facet_num", nearest_facet_num),
            pl.Series("face_center_from_facet_center", distance)
        )

    @staticmethod
    def rasterize_facets(
            points: pl.DataFrame, tris: pl.DataFrame,
            chunk: QCubeChunk,
            colour_column_name: Callable[[str], str] = lambda face: f'{face}_ratio'):

        pd_verts: pd.DataFrame = (points
                                  .select([f'{chunk.face}_u', f'{chunk.face}_v'])
                                  .rename({f'{chunk.face}_u': 'x', f'{chunk.face}_v': 'y'})
                                  .to_pandas())

        pd_tris: pd.DataFrame = (tris
                                 .select(['0', '1', '2', colour_column_name(chunk.face)])
                                 .with_columns([
                                     pl.col('0').cast(pl.Int32),
                                     pl.col('1').cast(pl.Int32),
                                     pl.col('2').cast(pl.Int32),
                                     pl.col(
                                         colour_column_name(chunk.face)).cast(
                                         pl.Float64),
                                 ])
                                 .to_pandas())

        W, H = chunk.length, chunk.length
        cvs = ds.Canvas(
            plot_width=W, plot_height=H,
            x_range=chunk.x_range,
            y_range=chunk.y_range,
        )

        if pd_tris.shape[0] == 0:
            return np.zeros((W, H), dtype=np.float64)

        mesh = du.mesh(pd_verts, pd_tris)

        agg = cvs.trimesh(
            pd_verts,
            pd_tris,
            mesh=mesh,
            agg=ds.first(
                colour_column_name(chunk.face)),
            interp=False)

        return agg.astype('float64').values
