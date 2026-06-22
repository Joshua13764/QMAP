from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class PANToCubemap:
    @staticmethod
    def sample_face_roi_simple_super_sample(
        pan_img: NDArray,
        face: str,
        x_range: Tuple[float, float],  # [0,1] left → right
        y_range: Tuple[float, float],  # [0,1] top → bottom
        sample_resolution: Tuple[int, int],
        super_sample_factor: int = 1
    ) -> NDArray:
        """
        Returns cubemap face with attempt to have:
        - top-left pixel ≈ corner A
        - anticlockwise winding (A→B→C→D) when viewed top-left origin, y down
        """
        out_w, out_h = sample_resolution
        ss = max(1, super_sample_factor)
        ss_w, ss_h = out_w * ss, out_h * ss

        # Reference size - use larger dimension to preserve aspect
        ref_size = max(out_w, out_h) * ss

        # Super-sampled normalized coordinates [-1,1] across full face
        xs = (np.arange(ss_w, dtype=np.float32) + 0.5) / ref_size * 2 - 1
        ys = (np.arange(ss_h, dtype=np.float32) + 0.5) / ref_size * 2 - 1

        # Crop to requested region in [-1,1] space
        x_min, x_max = x_range
        y_min, y_max = y_range
        x_min_n = 2 * x_min - 1
        x_max_n = 2 * x_max - 1
        y_min_n = 2 * y_min - 1
        y_max_n = 2 * y_max - 1

        xs = xs * ((x_max_n - x_min_n) / 2) + ((x_min_n + x_max_n) / 2)
        ys = ys * ((y_max_n - y_min_n) / 2) + ((y_min_n + y_max_n) / 2)

        # **Important**: we do NOT flip ys here — we handle orientation later
        U, V = np.meshgrid(xs, ys)

        # ── Direction vectors (standard OpenGL convention) ────────────────
        if face == "posx":
            dir_x, dir_y, dir_z = 1, V, -U
        elif face == "negx":
            dir_x, dir_y, dir_z = -1, V, U
        elif face == "posy":
            dir_x, dir_y, dir_z = U, 1, -V
        elif face == "negy":
            dir_x, dir_y, dir_z = U, -1, V
        elif face == "posz":
            dir_x, dir_y, dir_z = U, V, 1
        elif face == "negz":
            dir_x, dir_y, dir_z = -U, V, -1
        else:
            raise ValueError("Unsupported face")

        # Normalize
        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        norm = np.maximum(norm, 1e-8)
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm

        # ── Equirectangular mapping ────────────────────────────────────────
        H, W = pan_img.shape[:2]
        lon = np.arctan2(dir_x, dir_z)
        lat = np.arcsin(dir_y)

        mapx = ((lon / (2 * np.pi)) + 0.5) * (W - 1)
        mapy = (0.5 - (lat / np.pi)) * (H - 1)

        mapx = (mapx % W).astype(np.float32)
        mapy = np.clip(mapy, 0, H - 1).astype(np.float32)

        # Sample at super-resolution
        result = cv2.remap(
            pan_img,
            mapx, mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )

        if ss > 1:
            result = cv2.resize(
                result, (out_w, out_h), interpolation=cv2.INTER_AREA)

        return result.astype(np.float64, copy=False)
