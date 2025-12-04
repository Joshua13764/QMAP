from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import polars as pl
from joblib import delayed
from numpy import float64
from numpy._typing._array_like import NDArray
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.file_storage_adapters.polars_obj_adapter_fast import \
    FSPolarsObjAdapterFast
from boulder_statistics.steps.PAN_to_LOD import PANToLOD
from boulder_statistics.steps.utils.cubemap_lod_base import CubeMapLodBase
from boulder_statistics.steps.utils.polars_3D_expressions import \
    Polars3DExpressions
from boulder_statistics.steps.utils.projection_plotting import \
    ProjectionPlotting
from boulder_statistics.task_step_base import TaskStepBase

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class LodNode(CubeMapLodBase):
    debug_mode: bool
    export_folder: Path

    def render_region(self, face: str,
                      target_width: int) -> FSPathLocalDisk:
        posX, posY, depth, _ = self.get_proportion_roi()
        roi, total = self.get_region(target_width)

        x_range: Tuple[float, float] = (posX, posX + depth)
        y_range: Tuple[float, float] = (posY, posY + depth)

        points, tris = self.img
        tris_filtered: pl.LazyFrame = Polars3DExpressions.filter_faces_for_rasterization(
            tris, face)

        relative_path: Path = Path(*self.src_file.path).parent / Path(
            f"{Path(*self.src_file.path).name} LAS_lod_extract", f"lod_{len(self.shape)}", f"{face}_{roi[0]}_{roi[1]}_{roi[2]}x{roi[3]}_of_{total}.npy")

        export_file = FSPathLocalDisk(
            path=relative_path.parts,
            markers=frozenset([FSMarkerString(value="ProjectModel_lod")]),
            root_path=self.export_folder.as_posix()
        )

        export_file.make_directory()
        if not (export_file.exists and self.skip_if_exists):
            arr: NDArray[float64] = ProjectionPlotting.rasterize_tris(
                points, tris, face, x_range, y_range, (target_width, target_width))

            FSEnvironment.save(arr, export_file, FSNumpyAdapter())

        if self.debug_mode:
            ProjectionPlotting.plot_debug_data(
                points, tris_filtered, face, x_range, y_range, export_file.actual_path, self.skip_if_exists)

        return export_file

    def get_proportion_roi(self) -> Tuple[float, float, float, float]:
        posX = sum(xb / (2 ** (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))
        posY = sum(yb / (2 ** (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))

        depth = 2 ** (-len(self.shape))

        return (posX, posY, depth, depth)


@dataclass(frozen=True)
class OBJToLAS(TaskStepBase):
    lod_res: int
    depth: int
    skip_if_exists: bool
    debug_mode: bool
    export_folder: str

    def run(self, env: FSEnvironment) -> FSEnvironment:

        paths: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: FSMarkerString("ProjectModel") in x.markers)

        for path in paths:
            self.project_model(path)

        return FSEnvironment.merge([env])

    def project_model(self, file: FSPathLocalDisk) -> List[FSPathLocalDisk]:

        lazy_file_data: tuple[pl.LazyFrame, pl.LazyFrame] = FSEnvironment.load(
            file, FSPolarsObjAdapterFast())

        lazy_file_data = Polars3DExpressions.process_mesh(*lazy_file_data)
        if self.debug_mode:
            lazy_file_data = Polars3DExpressions.process_extra_mesh_data(
                *lazy_file_data)

        export_groups: List[FSPathLocalDisk] = []

        for lod_depth in range(self.depth + 1):
            export_groups += ParallelPbar(f"rendering lod_depth {lod_depth} for model {file.actual_path.name}")(n_jobs=1)(
                delayed(
                    LodNode.render_on_all_faces)(
                    LodNode(
                        shape,
                        lazy_file_data,
                        file,
                        self.skip_if_exists,
                        self.debug_mode,
                        Path(self.export_folder)),
                    target_width=self.lod_res)
                for shape in PANToLOD.all_binaries(bits=2 * lod_depth)
            )

        return export_groups
