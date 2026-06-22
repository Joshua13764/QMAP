from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import polars as pl
from numpy.typing import NDArray
from polars import LazyFrame

from boulder_statistics.file_storage_adapters.inference_detection_adapter import \
    FSInferenceDetectionAdapter
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.lods.utils.image_detection_grade import \
    ImageDetectionGrade
from boulder_statistics.lods.utils.image_detection_grade_loaded import \
    ImageDetectionGradeLoaded
from boulder_statistics.steps.export_image_detection_grades_as_df import \
    ExportImageDetectionGradesDataAsDF

ArrayType = NDArray[Any]
LazyFrameAction = Callable[[], LazyFrame]
LazyFrameActionBatch = List[LazyFrameAction]


@dataclass(frozen=True, kw_only=True)
class ExportDetectionMaskImagesAsDF(ExportImageDetectionGradesDataAsDF):
    """To export the LOD detection masks as a polars importable datatype. Needs to be exported as ".parquet" type or array will not propagate correctly!
    """

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

                # Array
                "detection_mask": pl.Array(pl.Float32, loaded_grades[0].detection_data.mask_uint8.size),
                "detection_mask_width": pl.Int32,
                "detection_mask_height": pl.Int32
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

                # Array
                "array": [loaded_grade.detection_data.mask_uint8.ravel() for loaded_grade in loaded_grades],
                "detection_mask_width": [loaded_grade.detection_data.mask_uint8.shape[0] for loaded_grade in loaded_grades],
                "detection_mask_height": [loaded_grade.detection_data.mask_uint8.shape[1] for loaded_grade in loaded_grades]
            }, schema=schema)

        return action
