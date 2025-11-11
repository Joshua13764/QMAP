from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import polars as pl
from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from bennu_feature_extractor.step_base import StepBase
from joblib import delayed
from numpy import dtype, float64
from numpy._typing._array_like import NDArray
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor_PDS.file_storage_adapters.polars_obj_adapter import \
    FSPolarsObjAdapter
from bennu_feature_extractor_PDS.file_storage_adapters.tiff_adapter import \
    FSTiffAdapter
from bennu_feature_extractor_PDS.PAN_to_LOD import PANToLOD
from bennu_feature_extractor_PDS.utils.cubemap_lod_base import CubeMapLodBase
from bennu_feature_extractor_PDS.utils.polars_3D_expressions import \
    Polars3DExpressions
from bennu_feature_extractor_PDS.utils.projection_plotting import \
    ProjectionPlotting

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class LodNode(CubeMapLodBase):
    debug_mode: bool

    def render_region(self, face: str,
                      target_width: int) -> FSPathLocalDisk:
        posX, posY, depth, _ = self.get_proportion_roi()
        roi, total = self.get_region(target_width)

        x_range: Tuple[float, float] = (posX, posX + depth)
        y_range: Tuple[float, float] = (posY, posY + depth)

        points, tris = self.img
        tris_filtered: pl.DataFrame = Polars3DExpressions.filter_faces_for_rasterization(
            tris, face)

        relative_path: Path = Path(*self.src_file.path).parent / Path(
            f"{Path(*self.src_file.path).name} LAS_lod_extract", f"lod_{len(self.shape)}", f"{face}_{roi[0]}_{roi[1]}_{roi[2]}x{roi[3]}_of_{total}.png")

        export_file = FSPathLocalDisk(
            path=relative_path.parts,
            markers=frozenset([FSMarkerString(value="ProjectModel_lod")]),
            root_path=self.src_file.root_path
        )

        export_file.make_directory()
        if not (export_file.exists and self.skip_if_exists):
            arr: NDArray[float64] = ProjectionPlotting.rasterize_tris(
                points, tris, face, x_range, y_range, (target_width, target_width))

            FSEnvironment.save(arr, export_file, FSTiffAdapter())

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


@dataclass()
class OBJToLAS(StepBase):
    lod_res: int
    depth: int
    root_path: Path
    skip_if_exists: bool
    debug_mode: bool

    def run(self, env: FSEnvironment) -> FSEnvironment:

        paths: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: FSMarkerString("ProjectModel") in x.markers)

        for path in paths:
            self.project_model(path)

        return FSEnvironment.merge([env])

    def project_model(self, file: FSPathLocalDisk) -> List[FSPathLocalDisk]:

        fileData: tuple[pl.DataFrame, pl.DataFrame] = FSEnvironment.load(
            file, FSPolarsObjAdapter())

        fileData = Polars3DExpressions.process_mesh(*fileData)
        if self.debug_mode:
            fileData = Polars3DExpressions.process_extra_mesh_data(*fileData)

        export_groups = []
        for lod_depth in range(self.depth + 1):
            export_groups += ParallelPbar(f"rendering lod_depth {lod_depth} for model {file.actual_path.name}")(n_jobs=-1)(
                delayed(
                    LodNode.render_on_all_faces)(
                    LodNode(
                        shape,
                        fileData,
                        file,
                        self.skip_if_exists,
                        self.debug_mode),
                    target_width=self.lod_res)
                for shape in PANToLOD.all_binaries(bits=2 * lod_depth)
            )

        return export_groups
