from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import datashader as ds
import datashader.transfer_functions as tf
import datashader.utils as du
import pandas as pd
import polars as pl
from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase
from joblib import Parallel, delayed
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor_PDS.file_storage_adapters.polars_obj_adapter import \
    FSPolarsObjAdapter
from bennu_feature_extractor_PDS.file_storage_adapters.tiff_adapter import \
    FSTiffAdapter
from bennu_feature_extractor_PDS.PAN_to_LOD import PANToLOD
from bennu_feature_extractor_PDS.utils.cubemap_lod_base import CubeMapLodBase
from bennu_feature_extractor_PDS.utils.polars_3D_expressions import (
    FACES, Polars3DExpressions)

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class LodNode(CubeMapLodBase):

    def render_region(self, face: str,
                      target_width: int) -> FSPathLocalDisk:
        posX, posY, depth, _ = self.get_proportion_roi()
        roi, total = self.get_region(target_width)

        x_range: Tuple[float, float] = (posX, posX + depth)
        y_range: Tuple[float, float] = (posY, posY + depth)

        tile = OBJToLAS.rasterize_tris(
            *self.img, face, x_range, y_range, (target_width, target_width))

        relative_path: Path = Path(*self.src_file.path).parent / Path(f"{Path(*self.src_file.path).name} LAS_lod_extract", f"lod_{len(self.shape)}",
                                                                      f"{face}_{roi[0]}_{roi[1]}_{roi[2]}x{roi[3]}_of_{total}.png")

        export_file = FSPathLocalDisk(
            path=relative_path.parts,
            markers=frozenset([FSMarkerString(value="ProjectModel_lod")]),
            root_path=self.src_file.root_path
        )

        export_file.make_directory()
        if not (export_file.exists and self.skip_if_exists):
            FSEnvironment.save(tile, export_file, FSTiffAdapter())

        return export_file

    def get_proportion_roi(self) -> Tuple[float, float, float, float]:
        posX = sum(xb / (2 ** (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))
        posY = sum(yb / (2 ** (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))

        depth = 2 ** (-len(self.shape))

        return (posX, posY, depth, depth)


@dataclass()
class OBJToLAS(StepBase):
    lod_res: int
    depth: int
    root_path: Path
    skip_if_exists: bool

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

        img: Tuple[pl.DataFrame, pl.DataFrame] = (points, tris)

        for lod_depth in range(self.depth + 1):
            export_groups += ParallelPbar(f"rendering lod_depth {lod_depth}")(n_jobs=-1)(
                delayed(
                    LodNode.render_on_all_faces)(
                    LodNode(shape, img, file, self.skip_if_exists),
                    target_width=self.lod_res)
                for shape in PANToLOD.all_binaries(bits=2 * lod_depth)
            )

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
