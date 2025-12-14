from dataclasses import dataclass, field
from email.policy import default
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import polars as pl
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.polars_obj_adapter_fast import \
    FSPolarsObjAdapterFast
from boulder_statistics.lods.img_lod_position import ImgLODPosition
from boulder_statistics.lods.img_lod_tile import LODImageTile
from boulder_statistics.lods.lod_image_utils import LODImageUtils
from boulder_statistics.step_default_markers import StepDefaultMarkers
from boulder_statistics.steps.PAN_to_LOD import PANToLOD
from boulder_statistics.steps.utils.cubemap_lod_base import CubeMapLodBase
from boulder_statistics.steps.utils.cubemaps_shared import (FACES,
                                                            LazyFileData, Pair,
                                                            PairGroups)
from boulder_statistics.steps.utils.lod_from_projection_renderer import \
    LodFromProjectionRenderer
from boulder_statistics.steps.utils.polars_3D_expressions import \
    Polars3DExpressions
from boulder_statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class ExtractOBJAsCubemapLods(TaskStepBase, StepDefaultMarkers):
    lod_res: int
    depth: int
    skip_if_exists: bool
    debug_mode: bool
    export_folder: FSPathLocalDisk
    adapter: FSAdapterBase[NDArray[np.float64], FSPathLocalDisk]
    message_prefix_generator: Callable[[int, Path], str] = field(
        default=lambda depth, src_path: f"""Rendering features for lod_depth {depth} and model {
            src_path.name}"""
    )
    n_jobs: int = field(default=1)
    export_resolution: int = field(default=512)
    verbose: bool = field(default=False)

    @property
    def hashable(self) -> tuple[Any, ...]:
        return self.include_markers_in_hashable(
            self.depth, self.export_folder, self.lod_res)

    def run(self, env: FSEnvironment) -> FSEnvironment:

        paths: List[FSPathLocalDisk] = self.get_files_with_markers(env)

        output_paths: List[FSPathLocalDisk] = []

        for path in paths:
            output_paths += self.project_model(path)

        return FSEnvironment(tuple(output_paths))

    def project_model(self, file: FSPathLocalDisk) -> List[FSPathLocalDisk]:

        lazy_file_data: LazyFileData = FSEnvironment.load(
            file, FSPolarsObjAdapterFast())

        lazy_file_data = Polars3DExpressions.process_mesh_projection_scaling(
            *lazy_file_data)
        lazy_file_data = Polars3DExpressions.add_displacement_columns(
            *lazy_file_data)

        lod_from_projection_renderers: List[LodFromProjectionRenderer] = self.get_lod_from_projection_renderers(
            *lazy_file_data)

        result_tiles: List[LODImageTile[np.float64]] = self.run_in_parallel(
            function=LodFromProjectionRenderer.render_lod,
            inputs=lod_from_projection_renderers,
            message=f"""Rendering DIS for lod_depth {
                self.depth} and model {
                file.actual_path.name}""",
            n_jobs=self.n_jobs
        )

        return [tile.local_disk_save_path for tile in result_tiles]

    def get_lod_from_projection_renderers(
            self, points: pl.LazyFrame, tris: pl.LazyFrame) -> List[LodFromProjectionRenderer]:

        return [
            LodFromProjectionRenderer(self.output_markers, self.adapter,
                                      face, tile, points, tris, self.export_folder,
                                      resolution=self.export_resolution, verbose=self.verbose)
            for face in FACES
            for tile in LODImageUtils.get_all_lod_tiles(depth=self.depth)
        ]
