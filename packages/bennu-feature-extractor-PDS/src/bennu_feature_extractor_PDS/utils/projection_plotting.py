from pathlib import Path
from typing import Callable, Tuple

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

# Plot settings
matplotlib.use("Agg")
plt.style.use('science')
plt.rcParams["figure.figsize"] = (7, 7 * ((5**0.5 - 1) / 2))
DPI = 400
plt.rcParams["figure.dpi"] = 400
plt.ioff()


class ProjectionPlotting:

    @staticmethod
    def plot_debug_data(points: pl.DataFrame,
                        tris: pl.DataFrame, face: str, x_range: Tuple[float, float], y_range: Tuple[float, float], main_save_path: Path, skip_if_exists: bool):

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'{face}_cos(angle)',
                                     title=rf"Tri $\mathrm{{A}}\,\cos{{\theta}}$ relative to cubemap face {face}",
                                     fig_save_name_suffix="plot_distance_normalized",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=rf"Tri $\mathrm{{A}}\,\cos{{\theta}}$ relative to cubemap face {face}",
                                     skip_if_exists=skip_if_exists)

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'xyz_radius',
                                     title=f"Tri center distance relative to origin for face {face}",
                                     fig_save_name_suffix="plot_xyz_distance",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=rf"Distance of the unprojected tri center from the origin",
                                     skip_if_exists=skip_if_exists)

        ProjectionPlotting.plot_data(points, tris, face, x_range, y_range, main_save_path,
                                     colour_column_name=lambda face: f'radius_ratio',
                                     title=r"$r_{\text{xyz}} / r_{\text{projected}}$ for face " + face,
                                     fig_save_name_suffix="plot_distance_ratio",
                                     scaling_function=lambda img: img,
                                     colour_bar_title=r"$r_{\text{xyz}} / r_{\text{projected}}$",
                                     skip_if_exists=skip_if_exists)

    @staticmethod
    def plot_data(points: pl.DataFrame,
                  tris: pl.DataFrame, face: str, x_range: Tuple[float, float], y_range: Tuple[float, float], main_save_path: Path,
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
    def rasterize_tris(points: pl.DataFrame,
                       tris: pl.DataFrame, face: str, x_range=(0, 1), y_range=(0, 1), res=(1024, 1024), colour_column_name: Callable[[str], str] = lambda face: f'{face}_ratio') -> NDArray[np.float64]:

        pd_verts: pd.DataFrame = (points.select([f'{face}_u', f'{face}_v'])
                                  .rename({f'{face}_u': 'x', f'{face}_v': 'y'})
                                  .to_pandas())

        pd_tris: pd.DataFrame = (tris.select(['0', '1', '2', colour_column_name(face)])
                                 .with_columns([
                                     pl.col('0').cast(pl.Int32),
                                     pl.col('1').cast(pl.Int32),
                                     pl.col('2').cast(pl.Int32),
                                     pl.col(
                                         colour_column_name(face)).cast(
                                         pl.Float64),
                                 ])
                                 .to_pandas())

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
