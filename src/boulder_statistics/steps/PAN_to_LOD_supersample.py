from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import cv2
import numpy as np
from joblib import delayed
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.iio_adapter import FSIIOAdapter
from boulder_statistics.steps.utils.cubemaps_shared import FACES
from boulder_statistics.task_step_base import TaskStepBase

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class LodNode:
    # sequence of (xbit,ybit) pairs, one per level (depth)
    shape: Tuple[Pair, ...]
    sample_factor: int
    img: Any
    src_file: FSPathLocalDisk
    task: "PANToLODSuperSample"

    def get_total_width(self, target_width: int) -> int:
        depth = len(self.shape)
        return target_width * (1 << depth)  # target * 2**depth

    def get_region(self, target_width: int):
        total = self.get_total_width(target_width)
        # step at level i (0-based) is total / 2**(i+1)
        posX = sum(xb * (total >> (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))
        posY = sum(yb * (total >> (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))
        roi = (int(posX), int(posY), int(target_width), int(target_width))
        return roi, total

    def render_region(self, face: str,
                      target_width: int) -> FSPathLocalDisk:
        roi, total = self.get_region(target_width)

        # total acts as the face resolution in mapping
        tile = PANToLODSuperSample.super_sample_face_roi(
            self.img, face, total, *roi, sample_factor=self.sample_factor)

        relative_path: Path = Path(*self.src_file.path).parent / Path(f"{self.task.extract_folder_prefix} {Path(*self.src_file.path).stem}", f"lod_{len(self.shape)}",
                                                                      f"{face}_{roi[0] // self.sample_factor}_{roi[1] // self.sample_factor}_{roi[2] // self.sample_factor}x{roi[3] // self.sample_factor}_of_{total // self.sample_factor}")

        export_file = FSPathLocalDisk(
            path=relative_path.parts,
            markers=self.task.export_markers,
            root_path=self.task.root_path.as_posix()
        )

        export_file.make_directory()
        if not (export_file.exists and self.task.skip_if_exists):
            FSEnvironment.save(tile, export_file, self.task.export_adapter)

        return export_file

    def render_on_all_faces(self, target_width: int) -> List[FSPathLocalDisk]:
        export_files: List[FSPathLocalDisk] = []

        for face in FACES:
            export_files.append(self.render_region(face, target_width))

        return export_files


@dataclass(frozen=True)
class PANToLODSuperSample(TaskStepBase):
    root_path: Path

    lod_depth: int = field(default_factory=lambda: 5)

    extract_folder_prefix: str = field(
        default_factory=lambda: "PAN_lod_extract")

    skip_if_exists: bool = field(default_factory=lambda: True)

    lod_res: int = field(default_factory=lambda: 512)

    import_markers: tuple[FSMarkerBase, ...] = field(
        default_factory=lambda: ())

    export_adapter: FSAdapterBase[NDArray[Any], FSPathLocalDisk] = field(
        default_factory=lambda: FSIIOAdapter(add_file_extension=".tif"))

    export_markers: tuple[FSMarkerBase, ...] = field(default_factory=lambda: (
        FSMarkerString(value="PAN_lod"), FSMarkerString(value="InferableImage")))

    supersample_factor: int = field(default_factory=lambda: 4)

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.root_path, self.lod_depth, self.extract_folder_prefix,
                self.lod_res, self.import_markers, self.export_adapter, self.export_markers,
                self.supersample_factor)

    def run(self, env: FSEnvironment) -> FSEnvironment:

        pan_src_files: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: ".tif" in x.actual_path.name and (
                set(self.import_markers).isdisjoint(x.markers) == False)
        )

        exports: List[FSPathLocalDisk] = []

        for file in pan_src_files:
            exports += self.render_lods_from_img(
                file,
                img=FSEnvironment.load(file, FSIIOAdapter())
            )

        return FSEnvironment.merge([env, FSEnvironment((exports,))])

    def render_lods_from_img(self, src_file: FSPathLocalDisk,
                             img: Any) -> List[FSPathLocalDisk]:

        export_groups: Any = ParallelPbar(f"rendering lods", unit=" 6 face imgs")(n_jobs=-1)(
            delayed(
                LodNode.render_on_all_faces)(
                LodNode(shape, self.supersample_factor, img, src_file, self),
                target_width=self.lod_res * self.supersample_factor)
            for lod_depth in range(self.lod_depth) for shape in self.all_binaries(bits=2 * lod_depth)
        )

        return [x for sub in export_groups for x in sub]

    @staticmethod
    def super_sample_face_roi(e_img: Any, face: str, face_w: int,
                              x0: int, y0: int, w: int, h: int, sample_factor: int):
        roi = PANToLODSuperSample.sample_face_roi(
            e_img, face, face_w, x0, y0, w, h)

        return cv2.resize(roi, (w // sample_factor, h //
                          sample_factor), interpolation=cv2.INTER_AREA)

    @staticmethod
    def sample_face_roi(e_img: Any, face: str, face_w: int,
                        x0: int, y0: int, w: int, h: int):
        H, W = e_img.shape[:2]

        # pixel centers for ROI in face space → [-1, 1]
        xs = (np.arange(x0, x0 + w, dtype=np.float32) + 0.5) / face_w * 2.0 - 1.0
        ys = (np.arange(y0, y0 + h, dtype=np.float32) + 0.5) / face_w * 2.0 - 1.0
        ys = ys[::-1]  # make top row +V
        U, V = np.meshgrid(xs, ys)

        # cube directions
        if face == "posx":
            x, y, z = np.ones_like(U), V, -U
        elif face == "negx":
            x, y, z = -np.ones_like(U), V, U
        elif face == "posy":
            x, y, z = U, np.ones_like(U), -V
        elif face == "negy":
            x, y, z = U, -np.ones_like(U), V
        elif face == "posz":
            x, y, z = U, V, np.ones_like(U)
        elif face == "negz":
            x, y, z = -U, V, -np.ones_like(U)
        else:
            raise ValueError("face must be posx/negx/posy/negy/posz/negz")

        # normalize & map to equirect (lon/lat → pixels)
        L = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-8)
        x, y, z = x / L, y / L, z / L
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -1.0, 1.0))

        mapx = (lon / (2 * np.pi) + 0.5) * W
        mapy = (0.5 - (lat / np.pi)) * H
        mapx = (mapx % W).astype(np.float32)         # wrap horizontally
        mapy = np.clip(mapy, 0, H - 1).astype(np.float32)  # clamp vertically

        # sample just the ROI
        tile = cv2.remap(
            e_img,
            mapx,
            mapy,
            interpolation=cv2.INTER_AREA,
            borderMode=cv2.BORDER_WRAP)
        return tile  # shape (h, w, C)

    @staticmethod
    def all_binaries(bits: int) -> Iterator[PairGroups]:
        """Yield tuples of (xbit,ybit) pairs. 'bits' must be even."""
        if bits % 2:
            raise ValueError("bits must be even (2 * depth).")
        for b in product('01', repeat=bits):
            it = iter(map(int, b))
            yield tuple(zip(it, it))  # ((xb0,yb0), (xb1,yb1), ...)

    @staticmethod
    def depth_from_sizes(face_w: int, leaf: int) -> int:
        """Depth s.t. (2**depth) * leaf >= face_w (covers non powers of two)."""
        ratio = max(1, int(np.ceil(face_w / leaf)))
        # round up to next power of two
        pow2 = int(2 ** int(np.ceil(np.log2(ratio))))
        return int(np.log2(pow2))
