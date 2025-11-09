from pathlib import Path
from typing import List

import datashader as ds
import datashader.transfer_functions as tf
import datashader.utils as du
import pandas as pd
import polars as pl
from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase
from colorcet import fire

from bennu_feature_extractor_PDS.file_storage_adapters.polars_obj_adapter import \
    FSPolarsObjAdapter
from bennu_feature_extractor_PDS.file_storage_adapters.trimesh_obj_adapter import \
    FSTrimeshAdapter
from bennu_feature_extractor_PDS.utils.polars_3D_expressions import (
    FACES, Polars3DExpressions)


class OBJToLAS(StepBase):
    root_path: Path

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: FSMarkerString("ProjectModel") in x.markers)

        return FSEnvironment.merge([env])

    def project_model(self, file: FSPathLocalDisk) -> List[FSPathLocalDisk]:

        fileData: tuple[pl.DataFrame, pl.DataFrame] = FSEnvironment.load(
            file, FSPolarsObjAdapter())

        points: pl.DataFrame = fileData[0]
        tris: pl.DataFrame = fileData[1]

        Polars3DExpressions.process_mesh(points, tris)

        for face in FACES:
            rasterized_array = self.rasterize_tris(points, tris, face)

            rasterized_array.save

    @staticmethod
    def rasterize_tris(points: pl.DataFrame,
                       tris: pl.DataFrame, face: str, x_range=(0, 1), y_range=(0, 1), res=(1024, 1024)):

        pd_verts: pd.DataFrame = (points.select([f'{face}_u', f'{face}_v'])
                                  .rename({f'{face}_u': 'x', f'{face}_v': 'y'})
                                  .to_pandas())

        pd_tris: pd.DataFrame = (tris.select(['0', '1', '2', f'{face}_ratio'])
                                 .with_columns([
                                     pl.col('0').cast(pl.Int32),
                                     pl.col('1').cast(pl.Int32),
                                     pl.col('2').cast(pl.Int32),
                                     pl.col(f'{face}_ratio').cast(pl.Float64),
                                 ])
                                 .to_pandas())

        mesh = du.mesh(pd_verts, pd_tris)

        tifffile.imwrite('tri_ratio.tiff', arr)

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
                f'{face}_ratio'),
            interp=False)

        return agg.astype('float64').values
