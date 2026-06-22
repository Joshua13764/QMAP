from dataclasses import dataclass
from typing import List, Tuple

from boulder_statistics.lods.utils.image_detection_grade import \
    ImageDetectionGrade


@dataclass
class ImageDetectionGrades():
    grades: Tuple[ImageDetectionGrade, ...]
