from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from polars import LazyFrame

from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.inference_detection_adapter import \
    FSInferenceDetectionAdapter
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.lods.utils.image_detection_grade import \
    ImageDetectionGrade
from boulder_statistics.lods.utils.image_detection_grade_loaded import \
    ImageDetectionGradeLoaded
from boulder_statistics.lods.utils.image_detection_grades import \
    ImageDetectionGrades
from boulder_statistics.steps.base.one_to_many_step_base import \
    OneToManyStepBase
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.base.output_adapter_step_base import \
    OutputAdapterStepBase

ArrayType = NDArray[Any]
LazyFrameAction = Callable[[], LazyFrame]
LazyFrameActionBatch = List[LazyFrameAction]


@dataclass(frozen=True, kw_only=True)
class ExportBoulderNetInferencesAsDF(
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

    def get_lazy_frame_from_batch(
            self, batch: List[ImageDetectionGrade]) -> LazyFrameAction:

        def action() -> LazyFrame:
            first_grade: ImageDetectionGrade = batch[0]
            loaded_grades: List[ImageDetectionGradeLoaded] = ImageDetectionGradeLoaded.all_from_detection(
                image_path=first_grade.image_path,
                detections_path=first_grade.detections_path,
                image_adapter=FSNumpyAdapter(),
                detection_adapter=FSInferenceDetectionAdapter())

            return LazyFrame({
                # Position
                "tile_face": [loaded_grade.position.face for loaded_grade in loaded_grades],
                "tile_reciprocal_length": [loaded_grade.position.reciprocal_length for loaded_grade in loaded_grades],
                "tile_reciprocal_area": [loaded_grade.position.reciprocal_area for loaded_grade in loaded_grades],
                "tile_x_min": [loaded_grade.position.x_range[0] for loaded_grade in loaded_grades],
                "tile_x_max": [loaded_grade.position.x_range[1] for loaded_grade in loaded_grades],
                "tile_y_min": [loaded_grade.position.y_range[0] for loaded_grade in loaded_grades],
                "tile_y_max": [loaded_grade.position.y_range[1] for loaded_grade in loaded_grades],

                # Detection data
                "relative_bounding_box_x_min": [loaded_grade.detection_data.box_xyxy[0] for loaded_grade in loaded_grades],
                "relative_bounding_box_y_min": [loaded_grade.detection_data.box_xyxy[1] for loaded_grade in loaded_grades],
                "relative_bounding_box_x_max": [loaded_grade.detection_data.box_xyxy[2] for loaded_grade in loaded_grades],
                "relative_bounding_box_y_max": [loaded_grade.detection_data.box_xyxy[3] for loaded_grade in loaded_grades],
                "BoulderNet_confidence": [loaded_grade.detection_data.score for loaded_grade in loaded_grades],

                # Array statistics
                "mean": [loaded_grade.image_array.mean() for loaded_grade in loaded_grades],
                "sum": [loaded_grade.image_array.sum() for loaded_grade in loaded_grades],
                "median": [np.median(loaded_grade.image_array) for loaded_grade in loaded_grades],
                "tile_x_shape": [loaded_grade.image_array.shape[0] for loaded_grade in loaded_grades],
                "tile_y_shape": [loaded_grade.image_array.shape[1] for loaded_grade in loaded_grades],

                # LAS statistics needed here !!!
            })

        return action
