from pathlib import Path
from typing import List

import polars as pl
from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase

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

        points = points.with_columns(
            Polars3DExpressions.get_project_points_expression())
