from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import polars as pl
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
from boulder_statistics.steps.export_image_detection_grades_as_df import \
    ExportImageDetectionGradesDataAsDF

ArrayType = NDArray[Any]
LazyFrameAction = Callable[[], LazyFrame]
LazyFrameActionBatch = List[LazyFrameAction]


@dataclass(frozen=True, kw_only=True)
class ExportFullDataPack(ExportImageDetectionGradesDataAsDF):

    def get_lazy_frame_from_batch(
            self, batch: List[ImageDetectionGrade]) -> LazyFrameAction:

        def action() -> LazyFrame:
            first_grade: ImageDetectionGrade = batch[0]
            loaded_grades: List[ImageDetectionGradeLoaded] = ImageDetectionGradeLoaded.all_from_detection(
                image_path=first_grade.image_path,
                detections_path=first_grade.detections_path,
                image_adapter=FSNumpyAdapter(),
                detection_adapter=FSInferenceDetectionAdapter())

            schema: Dict[str, Any] = {
                # Position
                "tile_face": pl.String,
                "tile_lod_number": pl.Int32,
                "tile_lod_code": pl.String,
                "tile_reciprocal_length": pl.Float64,
                "tile_reciprocal_area": pl.Float64,
                "tile_x_min": pl.Float64,
                "tile_x_max": pl.Float64,
                "tile_y_min": pl.Float64,
                "tile_y_max": pl.Float64,

                # Detection data
                "relative_bounding_box_x_min": pl.Float64,
                "relative_bounding_box_y_min": pl.Float64,
                "relative_bounding_box_x_max": pl.Float64,
                "relative_bounding_box_y_max": pl.Float64,
                "BoulderNet_confidence": pl.Float64,

                # Paths
                "image_tile_path": pl.String,
                "image_detections_path": pl.String,
            }

            return LazyFrame({
                # Position
                "tile_face": [loaded_grade.position.face for loaded_grade in loaded_grades],
                "tile_lod_number": [loaded_grade.position.lod_number for loaded_grade in loaded_grades],
                "tile_lod_code": [loaded_grade.position.string_rep for loaded_grade in loaded_grades],
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

                # Paths
                "image_tile_path": [first_grade.image_path.actual_path.as_posix() for loaded_grade in loaded_grades],
                "image_detections_path": [first_grade.detections_path.actual_path.as_posix() for loaded_grade in loaded_grades],
            }, schema=schema)

        return action
