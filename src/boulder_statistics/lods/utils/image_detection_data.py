from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


@dataclass
class ImageDetectionData():
    bounding_box_xyxy: NDArray[Any]
    confidence_score: float
    class_id: int
    boulder_mask_uint8: NDArray[Any]
