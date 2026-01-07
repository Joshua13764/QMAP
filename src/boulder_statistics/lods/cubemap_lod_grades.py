from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.img_lod_position import ImgLODPosition


@dataclass(frozen=True)
class CubemapLodGrades(ImgLODPosition):
    position: CubemapLodPosition
    grades:
