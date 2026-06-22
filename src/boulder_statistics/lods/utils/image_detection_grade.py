from dataclasses import dataclass, field

from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition


@dataclass(frozen=True, kw_only=True)
class ImageDetectionGrade():
    # Path of where the src image originally taken
    image_path: FSPathLocalDisk

    # Path of the scaling factor map
    LAS_factor_path: FSPathLocalDisk

    # Index of the path of the detection inferences (detections_path) as well
    # as the index of the detection in that inference set (detection_index)
    detections_path: FSPathLocalDisk
    detection_index: int

    @property
    def cubemap_position(self) -> CubemapLodPosition:
        """Finds the generated LOD position from the image path

        Returns:
            CubemapLodPosition: Generated lod position
        """
        return CubemapLodPosition.from_fs_path(
            self.image_path.actual_path)
