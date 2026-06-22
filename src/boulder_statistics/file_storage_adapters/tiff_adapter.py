from dataclasses import dataclass, field
from email.policy import default
from typing import Any, Callable

import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import Image

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSTiffAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):
    """Uses the tifffile module to read / write images with a wide range of precision (including float 64)"""
    standard_extension: str | None | bool = field(default="tif")
    export_as_jpeg: bool = field(default=False)
    export_function: Callable[[np.ndarray],
                              np.ndarray] = field(default=lambda x: x)

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return tifffile.imread(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        tifffile.imwrite(
            path.actual_path.as_posix(), obj.astype(np.float16), compression="zstd")

        if self.export_as_jpeg:

            obj = self.export_function(obj)

            # Mask finite values
            finite_mask = ~np.isnan(obj)

            if not np.any(finite_mask):
                # Case: all values are NaN
                img_8bit = np.zeros(obj.shape, dtype=np.uint8)

            else:
                finite_vals = obj[finite_mask]
                min_val = finite_vals.min()
                max_val = finite_vals.max()

                # Replace NaNs with min_val (or 0 — your choice)
                obj = np.nan_to_num(obj, nan=min_val)

                if max_val == min_val:
                    norm = np.zeros_like(obj, dtype=np.float32)
                else:
                    norm = (obj - min_val) / (max_val - min_val)

                img_8bit = (norm * 255).clip(0, 255).astype(np.uint8)

            # Save JPEG
            Image.fromarray(img_8bit).save(
                path.actual_path.with_suffix(".jpg"),
                format="JPEG",
                quality=95
            )
