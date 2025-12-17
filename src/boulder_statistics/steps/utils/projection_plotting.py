import operator
from functools import reduce
from math import e
from pathlib import Path
from typing import Callable, List, Tuple

import datashader as ds
import datashader.transfer_functions as tf
import datashader.utils as du
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scienceplots
from numpy.typing import NDArray

from boulder_statistics.steps.utils.polars_3D_expressions import (
    PROJECTED_POINT_ATTRS, VERT_ID_COLS)

# Plot settings
matplotlib.use("Agg")
plt.style.use('science')
plt.rcParams["figure.figsize"] = (7, 7 * ((5**0.5 - 1) / 2))
DPI = 400
plt.rcParams["figure.dpi"] = 400
plt.ioff()


class ProjectionPlotting:

    @staticmethod
    def plot_debug_data(points: pl.LazyFrame,
                        tris: pl.LazyFrame, face: str, x_range: Tuple[float, float], y_range: Tuple[float, float], main_save_path: Path, skip_if_exists: bool):

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'{face}_cos(angle)',
                                     title=rf"Tri $\mathrm{{A}}\,\cos{{\theta}}$ relative to cubemap face {face}",
                                     fig_save_name_suffix="plot_distance_normalized",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=rf"$\mathrm{{A}}\,\cos{{\theta}}$",
                                     skip_if_exists=skip_if_exists)

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'xyz_radius',
                                     title=f"Tri center distance to origin for face {face}",
                                     fig_save_name_suffix="plot_xyz_distance",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=r"$r_{\text{xyz}}$",
                                     skip_if_exists=skip_if_exists)

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'projected_radius',
                                     title=f"Projected Tri center distance to origin for face {face}",
                                     fig_save_name_suffix="plot_projected_distance",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=r"$r_{\text{projected}}$",
                                     skip_if_exists=skip_if_exists)

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'radius_ratio',
                                     title=r"$r_{\text{xyz}} / r_{\text{projected}}$ for face " + face,
                                     fig_save_name_suffix="plot_distance_ratio",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=r"$r_{\text{xyz}} / r_{\text{projected}}$",
                                     skip_if_exists=skip_if_exists)

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f"{face}_ratio",
                                     title=r"A_{\text{projected}} / A_{\text{xyz}} for face " + face,
                                     fig_save_name_suffix="plot_area_ratio",
                                     scaling_function=lambda img: 1 / img,
                                     colour_bar_title=r"$A_{\text{projected}} / A_{\text{xyz}}$",
                                     skip_if_exists=skip_if_exists)

    @staticmethod
    def plot_data(points: pl.LazyFrame,
                  tris: pl.LazyFrame, face: str, x_range: Tuple[float, float], y_range: Tuple[float, float], main_save_path: Path,
                  colour_column_name: Callable[[str], str], title: str, fig_save_name_suffix: str, scaling_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
                  colour_bar_title: str, skip_if_exists: bool) -> None:

        save_path: Path = main_save_path.with_name(
            main_save_path.stem + f"_{fig_save_name_suffix}")

        if save_path.exists and skip_if_exists:
            return

        img: NDArray[np.float64] = ProjectionPlotting.rasterize_tris(
            points, tris, face, x_range, y_range, res=(1024 * 4, 1024 * 4),
            colour_column_name=colour_column_name)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(scaling_function(img), origin="upper",
                       extent=[0, 1, 0, 1], aspect="equal")

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=colour_bar_title)
        fig.tight_layout()

        fig.savefig(
            save_path,
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white")

        plt.close(fig)

    @staticmethod
    def get_verts_filter(
            face: str, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> pl.Expr:
        verts_in_x_range: pl.Expr = (
            (pl.col(f"{face}_u") > x_range[0]) &
            (pl.col(f"{face}_u") < x_range[1])
        )

        verts_in_y_range: pl.Expr = (
            (pl.col(f"{face}_v") > y_range[0]) &
            (pl.col(f"{face}_v") < y_range[1])
        )

        return verts_in_x_range & verts_in_y_range

    @staticmethod
    def get_tris_filter(
            face, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> pl.Expr:

        conditions: List[pl.Expr] = [
            (
                (pl.col(f"{face}_u{vertex_index}") >= x_range[0]) &
                (pl.col(f"{face}_u{vertex_index}") <= x_range[1]) &

                (pl.col(f"{face}_v{vertex_index}") >= y_range[0]) &
                (pl.col(f"{face}_v{vertex_index}") <= y_range[1])
            )
            for vertex_index in VERT_ID_COLS
        ]

        return reduce(operator.or_, conditions)

    @staticmethod
    def get_lazy_filter_tris_not_in_view(
            face, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> pl.Expr:

        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)

        tri_min_x: pl.Expr = pl.min_horizontal(
            [f"{face}_u0", f"{face}_u1", f"{face}_u2"])
        tri_max_x: pl.Expr = pl.max_horizontal(
            [f"{face}_u0", f"{face}_u1", f"{face}_u2"])
        tri_min_y: pl.Expr = pl.min_horizontal(
            [f"{face}_v0", f"{face}_v1", f"{face}_v2"])
        tri_max_y: pl.Expr = pl.max_horizontal(
            [f"{face}_v0", f"{face}_v1", f"{face}_v2"])

        triangle_render_condition: pl.Expr = (
            (tri_min_x >= pl.lit(x_min)) &
            (tri_max_x <= pl.lit(x_max)) &
            (tri_min_y >= pl.lit(y_min)) &
            (tri_max_y <= pl.lit(y_max))
        )

        return triangle_render_condition

    @staticmethod
    def rasterize_tris(points: pl.LazyFrame,
                       tris: pl.LazyFrame, face: str, x_range=(0, 1), y_range=(0, 1), res=(1024, 1024), colour_column_name: Callable[[str], str] = lambda face: f'{face}_ratio') -> NDArray[np.float64]:

        pd_verts: pd.DataFrame = (points
                                  .select([f'{face}_u', f'{face}_v'])
                                  .rename({f'{face}_u': 'x', f'{face}_v': 'y'})
                                  .collect().to_pandas())

        pd_tris: pd.DataFrame = (tris
                                 .filter(ProjectionPlotting.get_lazy_filter_tris_not_in_view(face, x_range, y_range))
                                 .select(['0', '1', '2', colour_column_name(face)])
                                 .with_columns([
                                     pl.col('0').cast(pl.Int32),
                                     pl.col('1').cast(pl.Int32),
                                     pl.col('2').cast(pl.Int32),
                                     pl.col(
                                         colour_column_name(face)).cast(
                                         pl.Float64),
                                 ])
                                 .collect().to_pandas())

        mesh = du.mesh(pd_verts, pd_tris)

        W, H = res
        cvs = ds.Canvas(
            plot_width=W, plot_height=H,
            x_range=x_range,
            y_range=y_range,
        )

        agg = cvs.trimesh(
            pd_verts,
            pd_tris,
            mesh=mesh,
            agg=ds.first(
                colour_column_name(face)),
            interp=False)

        return agg.astype('float64').values
