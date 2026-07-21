from pathlib import Path
from time import perf_counter
from typing import Callable, List, Tuple

import datashader as ds
import datashader.utils as du
import igl
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import KDTree

from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData
from boulder_statistics.refinement_plus.qcube_chunk import QCubeChunk
from boulder_statistics.steps.utils.Better_polars_3D_expressions import (
    POINT_ATTRS, VERT_ID_COLS, BetterPolars3DExpressions)
from boulder_statistics.steps.utils.projection_plotting import \
    ProjectionPlotting


class FacetParser:

    @staticmethod
    def load_mesh(verts: np.ndarray,
                  faces: np.ndarray) -> Tuple[pl.DataFrame, pl.DataFrame]:
        # Meshes can be found here
        # https://sbnarchive.psi.edu/pds4/orex/orex.altimetry/data_derived_altimetry_global_models/global_digital_terrain_models/SPCv20/

        obj_data = PLOBJData(
            verts=pl.LazyFrame(
                verts, schema=POINT_ATTRS).with_row_index("vid"),
            tris=pl.LazyFrame(faces, schema=VERT_ID_COLS)
        )

        verts: pl.LazyFrame = obj_data.verts.with_columns(
            FacetParser.get_project_points_expression()
        ).with_columns(
            BetterPolars3DExpressions.get_project_points_calibration()
        ).with_columns(
            FacetParser.get_project_points_additional_calibration()
        )

        tris: pl.LazyFrame = BetterPolars3DExpressions._attach_points_to_tris(
            verts, obj_data.tris)
        tris: pl.LazyFrame = BetterPolars3DExpressions._add_area_and_ratio_columns(
            tris)
        tris: pl.LazyFrame = tris.with_columns(
            BetterPolars3DExpressions.get_mean_radius(),
            BetterPolars3DExpressions.get_mean_x().alias("x_tri_mean"),
            BetterPolars3DExpressions.get_mean_y().alias("y_tri_mean"),
            BetterPolars3DExpressions.get_mean_z().alias("z_tri_mean")
        )

        return verts.collect(), tris.collect().with_row_index("tri_num")

    @staticmethod
    def get_project_points_expression() -> List[pl.Expr]:
        """Projects directions (x, y, z) onto a cubemap (faces posx/negx/...) with
        UV in [0, 1] matching sample_face_roi. Calibration has been applied"""
        x, y, z = pl.col("x"), pl.col("y"), pl.col("z")
        sx, sy, sz = x.abs(), y.abs(), z.abs()

        return [
            (0.5 * ((z / sx) + 1.0)).alias("posx_u"),
            (0.5 * ((-y / sx) + 1.0)).alias("posx_v"),
            x.alias("posx_N"),

            (0.5 * ((-z / sx) + 1.0)).alias("negx_u"),
            (0.5 * ((-y / sx) + 1.0)).alias("negx_v"),
            (-x).alias("negx_N"),

            (0.5 * ((-x / sy) + 1.0)).alias("posy_u"),
            (0.5 * ((z / sy) + 1.0)).alias("posy_v"),
            y.alias("posy_N"),

            (0.5 * ((-x / sy) + 1.0)).alias("negy_u"),
            (0.5 * ((-z / sy) + 1.0)).alias("negy_v"),
            (-y).alias("negy_N"),

            (0.5 * ((x / sz) + 1.0)).alias("posz_u"),
            (0.5 * ((-y / sz) + 1.0)).alias("posz_v"),
            z.alias("posz_N"),

            (0.5 * ((-x / sz) + 1.0)).alias("negz_u"),
            (0.5 * ((-y / sz) + 1.0)).alias("negz_v"),
            (-z).alias("negz_N"),
        ]

    @staticmethod
    def get_project_points_additional_calibration() -> List[pl.Expr]:
        return [
            # posx = flip(flip(posx.T, axis=1), axis=0)
            (pl.col("posx_v")).alias("posx_u"),
            (1.0 - pl.col("posx_u")).alias("posx_v"),
            pl.col("posx_N").alias("posx_N"),

            # negx = flip(flip(negx.T, axis=1), axis=0)
            (pl.col("negx_v")).alias("negx_u"),
            (1.0 - pl.col("negx_u")).alias("negx_v"),
            pl.col("negx_N").alias("negx_N"),

            # posy = flip(posy.T, axis=1)
            (pl.col("posy_v")).alias("posy_u"),
            (pl.col("posy_u")).alias("posy_v"),
            pl.col("posy_N").alias("posy_N"),

            # negy = flip(negy.T, axis=1)
            (pl.col("negy_v")).alias("negy_u"),
            (pl.col("negy_u")).alias("negy_v"),
            pl.col("negy_N").alias("negy_N"),

            # posz = flip(flip(posz.T, axis=1), axis=0)
            (1.0 - pl.col("posz_v")).alias("posz_u"),
            (pl.col("posz_u")).alias("posz_v"),
            pl.col("posz_N").alias("posz_N"),

            # negz = flip(flip(negz.T, axis=1), axis=0)
            (1.0 - pl.col("negz_v")).alias("negz_u"),
            (pl.col("negz_u")).alias("negz_v"),
            pl.col("negz_N").alias("negz_N"),
        ]

    @staticmethod
    def join_with_facet_tri_data(
            facets: pl.DataFrame, tris: pl.DataFrame) -> pl.DataFrame:
        facet_nums_to_tri_nums_lookup: pl.DataFrame = FacetParser.get_facet_nums_to_tri_nums_lookup(
            facets, tris
        )

        facet_tri_joint_db: pl.DataFrame = tris.join(
            facet_nums_to_tri_nums_lookup,
            on="tri_num"
        ).join(
            facets,
            on="facet_num"
        )

        return facet_tri_joint_db

    @staticmethod
    def get_tri_chunk_tree_and_tri_ids(
            tris_df: pl.DataFrame) -> Tuple[KDTree, np.ndarray]:
        df: np.ndarray = tris_df.select(
            ["tri_num", "x_tri_mean", "y_tri_mean", "z_tri_mean"]
        ).to_numpy()

        tri_nums: np.ndarray = df[:, 0].astype(np.int32)
        tris_mean_xyz: np.ndarray = df[:, 1:].astype(np.float32)

        return KDTree(tris_mean_xyz), tri_nums

    @staticmethod
    def get_facet_positions_and_ids(
            facets: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_facet_data: np.ndarray = (
            facets
            .select(["facet_num", "x", "y", "z"])
            .to_numpy()
        )

        facet_nums: np.ndarray = df_facet_data[:, 0].astype(np.int32)
        facet_mean_xyz: np.ndarray = df_facet_data[:, 1:].astype(np.float32)

        return facet_mean_xyz, facet_nums

    @staticmethod
    def get_facet_nums_to_tri_nums_lookup(
            facets: pl.DataFrame, mesh_tris: pl.DataFrame) -> pl.DataFrame:

        facet_mean_xyz, facet_nums = FacetParser.get_facet_positions_and_ids(
            facets)
        tree, tri_nums = FacetParser.get_tri_chunk_tree_and_tri_ids(
            mesh_tris)

        distance, idx = tree.query(
            facet_mean_xyz,
            k=1,
            workers=-1,
        )

        assert np.all(idx < len(tri_nums)), (
            f"KDTree returned invalid indices: max idx={idx.max()}, "
            f"number of triangles={len(tri_nums)}"
        )

        return pl.DataFrame({
            "facet_num": facet_nums,
            "tri_num": tri_nums[idx],
            "associate_distance": distance
        })

    @staticmethod
    def rasterize_facets(
            points: pl.DataFrame, tris: pl.DataFrame,
            chunk: QCubeChunk):

        pd_verts: pd.DataFrame = (points
                                  .select([f'{chunk.face}_u', f'{chunk.face}_v'])
                                  .rename({f'{chunk.face}_u': 'x', f'{chunk.face}_v': 'y'})
                                  .to_pandas())

        pd_tris: pd.DataFrame = (tris
                                 #  .filter(ProjectionPlotting.get_lazy_filter_tris_not_in_view(chunk.face, chunk.x_range, chunk.y_range)
                                 .filter(ProjectionPlotting.get_lazy_filter_faces_for_rasterization_by_face(chunk.face))
                                 .select(['0', '1', '2', "tri_num"])
                                 .with_columns([
                                     pl.col('0').cast(pl.Int32),
                                     pl.col('1').cast(pl.Int32),
                                     pl.col('2').cast(pl.Int32),
                                     pl.col("tri_num").cast(pl.Float64),
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
            agg=ds.first("tri_num"),
            interp=False)

        return agg.values
