from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

from numpy.typing import NDArray

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_input import FSInput
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.adapter_custom_classes.inference_detection_data import \
    InferenceDetectionData
from boulder_statistics.file_storage_adapters.adapter_custom_classes.npz_feature_detection import \
    NpzFeatureDetection
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.fs_generic_cubemap_generator import \
    FSGenericCubemapGenerator
from boulder_statistics.lods.utils.image_detection_grade import \
    ImageDetectionGrade
from boulder_statistics.lods.utils.image_detection_grades import \
    ImageDetectionGrades
from boulder_statistics.steps.base.many_to_one_step_base import \
    SelectedToOneStepBase

ArrayType = NDArray[Any]
FSArrayCubemapGenerator = FSGenericCubemapGenerator[ArrayType]

FSInferencesCubemapGenerator = FSGenericCubemapGenerator[List[InferenceDetectionData]]


@dataclass(frozen=True, kw_only=True)
class SetupBoulderNetInferencesForGrading(
        SelectedToOneStepBase[ImageDetectionGrades]):

    lod_images_input: FSInput[FSArrayCubemapGenerator]
    lod_detections_input: FSInput[FSInferencesCubemapGenerator]
    collect_grades_workers: int = field(default=-1)

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.task_name,)

    def get_object_relative_export_path(
            self, env: FSEnvironment, output_object: ImageDetectionGrades) -> Tuple[str, ...]:
        return ("cubemap_grading",)

    def selected_objects_operation(self,
                                   env: FSEnvironment) -> ImageDetectionGrades:

        cubemap_images: FSArrayCubemapGenerator = self.lod_images_input.object(
            env)
        cubemap_inferences: FSInferencesCubemapGenerator = self.lod_detections_input.object(
            env)

        grades: List[Tuple[ImageDetectionGrade, ...]] = self.run_in_parallel(
            function=SetupBoulderNetInferencesForGrading.compile_get_grades_from_inferred_image(
                cubemap_images, cubemap_inferences),
            inputs=list(cubemap_inferences.tiles),
            message="Collecting grades",
            unit="tiles",
            n_jobs=self.collect_grades_workers,
        )

        grades_flattened: Tuple[ImageDetectionGrade, ...] = tuple(
            grade for grade_chunk in grades for grade in grade_chunk)

        return ImageDetectionGrades(grades=grades_flattened)

    @staticmethod
    def compile_get_grades_from_inferred_image(
            image_generator: FSArrayCubemapGenerator, inference_generator: FSInferencesCubemapGenerator) -> Callable[..., Tuple[ImageDetectionGrade, ...]]:
        return lambda tile: SetupBoulderNetInferencesForGrading.get_grades_from_inferred_image(
            tile, image_generator, inference_generator)

    @staticmethod
    def get_grades_from_inferred_image(tile: CubemapLodPosition, image_generator: FSArrayCubemapGenerator,
                                       inference_generator: FSInferencesCubemapGenerator) -> Tuple[ImageDetectionGrade, ...]:
        img_path: FSPathLocalDisk = image_generator.get_tile_path(tile)
        inference_path: FSPathLocalDisk = inference_generator.get_tile_path(
            tile)
        inferences_data: List[InferenceDetectionData] = inference_generator.get_lod_tile(
            tile)

        return tuple(
            ImageDetectionGrade(
                image_path=img_path,
                detections_path=inference_path,
                detection_index=inference_data_index,
            )
            for inference_data_index, inference_data in enumerate(inferences_data)
        )
