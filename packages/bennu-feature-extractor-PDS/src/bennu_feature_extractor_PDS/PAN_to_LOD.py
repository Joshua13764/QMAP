import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from sys import path
from typing import Any, Iterator, List, Tuple

import cv2
import numpy as np
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from bennu_feature_extractor.step_base import StepBase
from joblib import Parallel, delayed
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor_PDS.file_storage_adapters.iio_adapter import \
    FSIIOAdapter

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]

OUT_MARKERS: List[FSMarkerString] = [
    FSMarkerString(
        value="PAN_lod"), FSMarkerString(
            value="InferableImage")]


@dataclass(frozen=True, slots=True)
class LodNode:
    # sequence of (xbit,ybit) pairs, one per level (depth)
    shape: Tuple[Pair, ...]
    img: Any
    src_file: FSPathLocalDisk
    root_path: Path
    skip_if_exists: bool

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
        tile = PANToLOD.sample_face_roi(self.img, face, total, *roi)

        relative_path: Path = Path(*self.src_file.path).parent / Path(f"{Path(*self.src_file.path).name} lod_extract", f"lod_{len(self.shape)}",
                                                                      f"{face}_{roi[0]}_{roi[1]}_{roi[2]}x{roi[3]}_of_{total}.png")

        export_file = FSPathLocalDisk(
            path=relative_path.parts,
            markers=frozenset(OUT_MARKERS),
            root_path=self.root_path.as_posix()
        )

        export_file.make_directory()
        if not (export_file.exists and self.skip_if_exists):
            FSEnvironment.save(tile, export_file, FSIIOAdapter())

        return export_file

    def render_on_all_faces(self, target_width: int) -> List[FSPathLocalDisk]:
        faces = ["posx", "negx", "posy", "negy", "posz", "negz"]
        export_files: List[FSPathLocalDisk] = []

        for face in faces:
            export_files.append(self.render_region(face, target_width))

        return export_files


@dataclass(frozen=True)
class PANToLOD(StepBase):
    root_path: str
    lod_res: int
    skip_if_exists: bool

    def run(self, env: FSEnvironment) -> FSEnvironment:

        pan_src_files: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: ".tif" in x.actual_path.name)

        exports: List[FSPathLocalDisk] = []

        for file in pan_src_files:
            exports += self.render_lods_from_img(
                file,
                img=FSEnvironment.load(file, FSIIOAdapter())
            )

        return FSEnvironment.merge([env, FSEnvironment(frozenset(exports))])

    def render_lods_from_img(self, src_file: FSPathLocalDisk,
                             img: Any) -> List[FSPathLocalDisk]:

        H, W = img.shape[:2]
        face_w = int(2**np.ceil(np.log2(np.sqrt((W * H) / 6.0))))
        depth = self.depth_from_sizes(face_w, self.lod_res)

        export_groups: List[List[FSPathLocalDisk]] = []

        for lod_depth in range(depth + 1):
            export_groups += ParallelPbar(f"rendering lod_depth {lod_depth}")(n_jobs=-1)(
                delayed(
                    LodNode.render_on_all_faces)(
                    LodNode(
                        shape,
                        img,
                        src_file,
                        Path(self.root_path),
                        self.skip_if_exists),
                    target_width=self.lod_res)
                for shape in self.all_binaries(bits=2 * lod_depth)
            )

        return [x for sub in export_groups for x in sub]

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
            interpolation=cv2.INTER_LINEAR,
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
