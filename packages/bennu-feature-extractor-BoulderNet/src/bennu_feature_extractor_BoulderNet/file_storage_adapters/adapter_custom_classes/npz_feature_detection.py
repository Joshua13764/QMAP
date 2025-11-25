from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class NpzFeatureDetection():
    face: str
    relative_offset: NDArray[Any]
    relative_scale: float
    box_xyxy: NDArray[Any]
    score: float
    class_id: int
    mask_uint8: NDArray[Any]

    def get_area_fixed_weight(self, per_pixel_weight: float = 1) -> float:
        return np.sum(self.mask_uint8) * per_pixel_weight

    def get_area_variable_weight(self, weight_map: NDArray[Any]) -> float:
        return np.sum(self.mask_uint8 * weight_map)
