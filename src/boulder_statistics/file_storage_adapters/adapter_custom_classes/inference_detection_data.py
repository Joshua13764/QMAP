from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class InferenceDetectionData():
    box_xyxy: NDArray[Any]
    score: float
    class_id: int
    mask_uint8: NDArray[Any]
