from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class CubeMapLodBase(ABC):

    shape: Tuple[Pair, ...]
    img: Any
    src_file: FSPathLocalDisk
    skip_if_exists: bool

    def get_total_width(self, target_width: int) -> int:
        depth: int = len(self.shape)
        return target_width * (1 << depth)  # target * 2**depth

    def get_region(self, target_width: int):
        total = self.get_total_width(target_width)
        # step at level i (0-based) is total / 2**(i+1)
        posX = sum(xb * (total >> (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))
        posY = sum(yb * (total >> (i + 1))
                   for i, (xb, yb) in enumerate(self.shape))
        roi = (int(posX), int(posY), int(target_width), int(target_width))
        return roi, total

    @abstractmethod
    def render_region(self, face: str,
                      target_width: int) -> FSPathLocalDisk:
        ...

    def render_on_all_faces(self, target_width: int) -> List[FSPathLocalDisk]:
        faces = ["posx", "negx", "posy", "negy", "posz", "negz"]
        export_files: List[FSPathLocalDisk] = []

        for face in faces:
            export_files.append(self.render_region(face, target_width))

        return export_files
