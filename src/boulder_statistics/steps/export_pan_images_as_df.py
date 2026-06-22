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
class ExportPANImagesAsDF(ExportImageDetectionGradesDataAsDF):
    """To export the LOD images as a polars importable datatype. Needs to be exported as ".parquet" type or array will not propagate correctly!
    """

    def get_lazy_frame_from_batch(
            self, batch: List[ImageDetectionGrade]) -> LazyFrameAction:

        def action() -> LazyFrame:
            first_grade: ImageDetectionGrade = batch[0]
            loaded_first_grade: ImageDetectionGradeLoaded = ImageDetectionGradeLoaded.all_from_detection(
                image_path=first_grade.image_path,
                detections_path=first_grade.detections_path,
                image_adapter=FSNumpyAdapter(),
                detection_adapter=FSInferenceDetectionAdapter())[0]

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

                # Array
                "array": pl.Array(pl.Float32, loaded_first_grade.image_array.size),
                "array_width": pl.Int32,
                "array_height": pl.Int32
            }

            return LazyFrame({
                # Position
                "tile_face": [loaded_first_grade.position.face],
                "tile_lod_number": [loaded_first_grade.position.lod_number],
                "tile_lod_code": [loaded_first_grade.position.string_rep],
                "tile_reciprocal_length": [loaded_first_grade.position.reciprocal_length],
                "tile_reciprocal_area": [loaded_first_grade.position.reciprocal_area],
                "tile_x_min": [loaded_first_grade.position.x_range[0]],
                "tile_x_max": [loaded_first_grade.position.x_range[1]],
                "tile_y_min": [loaded_first_grade.position.y_range[0]],
                "tile_y_max": [loaded_first_grade.position.y_range[1]],

                # Array
                "array": [loaded_first_grade.image_array.ravel()],
                "array_width": [loaded_first_grade.image_array.shape[0]],
                "array_height": [loaded_first_grade.image_array.shape[1]]
            }, schema=schema)

        return action
