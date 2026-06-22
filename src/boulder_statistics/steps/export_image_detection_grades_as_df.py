from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from numpy.typing import NDArray
from polars import LazyFrame

from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.utils.image_detection_grade import \
    ImageDetectionGrade
from boulder_statistics.lods.utils.image_detection_grades import \
    ImageDetectionGrades
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase

ArrayType = NDArray[Any]
LazyFrameAction = Callable[[], LazyFrame]
LazyFrameActionBatch = List[LazyFrameAction]


@dataclass(frozen=True, kw_only=True)
class ExportImageDetectionGradesDataAsDF(
        OneToOneStepBase[ImageDetectionGrades, LazyFrameActionBatch]):

    @property
    def hashable(self) -> Tuple[Any, ...]:
        return (self.task_name,)

    def get_object_relative_export_path(
            self, input_object: ImageDetectionGrades, output_object: LazyFrameActionBatch) -> Tuple[str, ...]:
        return ("export", "")

    def object_operation(self,
                         input_object: ImageDetectionGrades) -> LazyFrameActionBatch:

        output_batches: List[LazyFrameAction] = [self.get_lazy_frame_from_batch(batch) for batch in self.batch_grades(
            input_object)]

        return output_batches

    def batch_grades(
            self, grades: ImageDetectionGrades) -> List[List[ImageDetectionGrade]]:

        grouped_by_detect_path: Dict[FSPathLocalDisk,
                                     List[ImageDetectionGrade]] = {}

        for grade in grades.grades:

            if grade.detections_path in grouped_by_detect_path:
                grouped_by_detect_path[grade.detections_path].append(grade)

            else:
                grouped_by_detect_path[grade.detections_path] = [grade]

        return list(grouped_by_detect_path.values())

    @abstractmethod
    def get_lazy_frame_from_batch(
            self, batch: List[ImageDetectionGrade]) -> LazyFrameAction:
        ...
