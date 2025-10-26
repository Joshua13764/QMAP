from __future__ import annotations

import numpy as np
import cv2
import pds4_tools
from typing import Any, Mapping, Sequence, Protocol
from numpy.typing import NDArray

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

# ---- Types
class ArrayStructure(Protocol):
    data: NDArray[Any]
    meta_data: Mapping[str, Any]
    def as_masked(self) -> "ArrayStructure": ...

StructureList = Sequence[ArrayStructure]


class EnvFilePDS4XML(EnvFileBase):
    def read(self) -> tuple[StructureList, NDArray[Any]]:
        sl: StructureList = pds4_tools.read(self.actual_path.as_posix(), lazy_load=True)
        img = self._to_viewable(sl)
        return sl, img

    def write(self, data: Any) -> None:
        raise NotImplementedError("Writing PDS4 files is not supported yet.")

    @staticmethod
    def _to_viewable(sl: StructureList) -> NDArray[Any]:
        # 1) pick first Array_* with NumPy data
        s = next((x for x in sl if hasattr(x, "data") and isinstance(x.data, np.ndarray)), None)
        if s is None:
            raise ValueError("No Array_* structure with NumPy data found.")

        # 2) use masked data if available (fill masked with 0), keep original dtype/range
        data = getattr(s.as_masked(), "data", s.data)
        if np.ma.isMaskedArray(data):
            data = data.filled(0)
        img = np.squeeze(np.asarray(data))

        # 3) reorder axes to (Line, Sample, *rest) if names exist
        aa = getattr(s, "meta_data", {}).get("Axis_Array")
        if aa:
            names = [aa["axis_name"]] if isinstance(aa, dict) else [d["axis_name"] for d in aa]
            if "Line" in names and "Sample" in names:
                vi, hi = names.index("Line"), names.index("Sample")
                if img.ndim == 2:
                    img = np.transpose(img, (vi, hi))
                elif img.ndim >= 3:
                    rest = [i for i in range(img.ndim) if i not in (vi, hi)]
                    img = np.transpose(img, (vi, hi, *rest))

        return img
    
    def show(self) -> None:
        _, img = self.read()  # already squeezed + (Line, Sample, ...)
        disp = img

        # If float, cv2.imshow expects values in [0, 1]
        if np.issubdtype(disp.dtype, np.floating):
            m, M = float(np.nanmin(disp)), float(np.nanmax(disp))
            disp = ((disp - m) / (M - m) if M > m else np.zeros_like(disp)).astype(np.float32)

        # If clearly 3-channel last, assume RGB -> BGR for OpenCV
        if disp.ndim == 3 and disp.shape[-1] == 3:
            disp = disp[..., ::-1]

        cv2.imshow("PDS4 Image", np.ascontiguousarray(disp))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

