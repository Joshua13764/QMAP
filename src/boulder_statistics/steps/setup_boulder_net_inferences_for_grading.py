from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_input import FSInput
from boulder_statistics.environment_tools.fs_object import FSObject
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import FSPathLocalDisk
from boulder_statistics.lods.utils.image_detection_data import ImageDetectionData
from boulder_statistics.lods.utils.image_detection_grades import \
    ImageDetectionGrades
from boulder_statistics.steps.base.many_to_one_step_base import \
    SelectedToOneStepBase
from boulder_statistics.steps.base.one_to_many_step_base import \

ArrayType = NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class SetupBoulderNetInferencesForGrading(
        SelectedToOneStepBase[ImageDetectionGrades]):

    lod_images_input: FSInput[ArrayType]
    lod_detections_input: FSInput[ImageDetectionData]

    def get_object_relative_export_path(
            self, env: FSEnvironment, output_object: ImageDetectionGrades) -> Tuple[str, ...]:
        ...

    def selected_objects_operation(self,
                                   env: FSEnvironment) -> ImageDetectionGrades:

        lod_images: FSObject[ArrayType,
                             FSPathLocalDisk] = self.lod_images_input.get_fs_object(env)
        lod_detections_input: FSObject[ImageDetectionData,
                                       FSPathLocalDisk] = self.lod_detections_input.get_fs_object(env)
