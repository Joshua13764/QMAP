from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Any
from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

@dataclass
class EnvFilePNG(EnvFileBase):
    overwrite_allowed: bool = True
    
    def read(self) -> NDArray[Any]:
        img = cv2.imread(self.actual_path.as_posix(), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read PNG: {self.actual_path}")
        return img

    def write(self, data: NDArray[Any]) -> None:
        self.logger.debug(f"Writing PNG to {self.actual_path}")

        if not self.overwrite_allowed and self.actual_path.exists():
            self.logger.warning(f"File already exists and overwire is not allowed: {self.actual_path}")
            return

        img, params = self._to_png_data(data)
        self.actual_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(self.actual_path.as_posix(), img)

        if not ok:
            raise IOError(f"Failed to write PNG: {self.actual_path}")

    @staticmethod
    def _to_png_data(arr: NDArray[Any]) -> tuple[NDArray[np.uint8] | NDArray[np.uint16], list[int]]:
        x = np.asarray(arr)
        x = np.squeeze(x)
        if x.dtype in (np.uint8, np.uint16):
            img = x
        else:
            xf = x.astype(np.float32, copy=False)
            m, M = float(np.nanmin(xf)), float(np.nanmax(xf))
            xf = (xf - m) / (M - m) if M > m else np.zeros_like(xf, dtype=np.float32)
            img = (xf * 255.0).clip(0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., ::-1]
        img = np.ascontiguousarray(img)
        params: list[int] = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        return img, params
