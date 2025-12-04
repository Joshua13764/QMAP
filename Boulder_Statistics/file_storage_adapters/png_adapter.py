from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPNGAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        img = cv2.imread(path.actual_path.as_posix(), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(
                f"Could not read PNG: {
                    path.actual_path.as_posix()}")
        return img

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        path.make_directory()
        write_state: bool = cv2.imwrite(
            path.actual_path.as_posix(),
            FSPNGAdapter._to_imwrite_dtype(obj))
        if not write_state:

            raise IOError(f"Failed to write PNG: {path.actual_path}")

    @staticmethod
    def _to_imwrite_dtype(a: NDArray[np.generic]) -> NDArray[np.generic]:
        """Make `a` compatible with cv2.imwrite (uint8/uint16, valid shape)."""
        # Squeeze trailing singleton channel: (H,W,1) -> (H,W)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]

        # If already uint8/uint16, keep as-is
        if a.dtype == np.uint8 or a.dtype == np.uint16:
            return np.ascontiguousarray(a)

        # Floats: assume 0..1 → uint8; otherwise normalize to 16-bit
        if np.issubdtype(a.dtype, np.floating):
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            vmin = float(a.min())
            vmax = float(a.max())
            if vmin >= 0.0 and vmax <= 1.0:
                a = (a * 255.0).round().astype(np.uint8)
            else:
                # normalize to 0..65535 for more depth
                if vmax == vmin:
                    a = np.zeros_like(a, dtype=np.uint16)
                else:
                    a = ((a - vmin) * (65535.0 / (vmax - vmin))
                         ).round().astype(np.uint16)
            return np.ascontiguousarray(a)

        # Other integer types: clip/cast to 8 or 16 bit
        if np.issubdtype(a.dtype, np.signedinteger) or np.issubdtype(
                a.dtype, np.unsignedinteger):
            amin = int(a.min())
            amax = int(a.max())
            if 0 <= amin and amax <= 255:
                return np.ascontiguousarray(a.astype(np.uint8))
            a = np.clip(a, 0, 65535).astype(np.uint16)
            return np.ascontiguousarray(a)

        # Fallback: normalize to 8-bit
        a = a.astype(np.float32)
        a -= a.min()
        m = float(a.max())
        if m > 0:
            a *= 255.0 / m
        return np.ascontiguousarray(a.astype(np.uint8))
