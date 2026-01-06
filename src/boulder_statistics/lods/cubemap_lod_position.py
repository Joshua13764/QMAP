from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.img_lod_position import ImgLODPosition


@dataclass(frozen=True)
class CubemapLodPosition(ImgLODPosition):
    face: str

    def get_fs_path(self, root_path: FSPathLocalDisk,
                    markers: Tuple[FSMarkerBase, ...], tile_shape: Tuple[int, ...]) -> FSPathLocalDisk:

        lod_str_rep: str = self.string_rep
        lod_number: int = self.lod_number

        rel_path: Path = Path(
            f"face {self.face}",
            f"lod {lod_number}",
            f"""lod tile ({lod_str_rep}) with_shape ({tile_shape})"""
        )

        local_disk_save_path: FSPathLocalDisk = root_path.copy_from_folder(
            rel_path, markers=markers)

        return local_disk_save_path

    @staticmethod
    def is_correct_path_format(path: Path) -> bool:

        raw_tile_name: str = path.stem
        raw_tile_lod_name: str = path.parent.stem
        raw_tile_face_name: str = path.parent.parent.stem

        raw_tile_name_check: bool = "lod tile (" in raw_tile_name and ") with_shape (" in raw_tile_name
        raw_tile_lod_name_check: bool = "lod " in raw_tile_lod_name
        raw_tile_face_name_check: bool = "face " in raw_tile_face_name

        return all(
            (raw_tile_name_check, raw_tile_lod_name_check,
             raw_tile_face_name_check)
        )

    @classmethod
    def from_fs_path(cls, path: Path) -> "CubemapLodPosition":

        raw_tile_name: str = path.stem
        raw_tile_lod_name: str = path.parent.stem
        raw_tile_face_name: str = path.parent.parent.stem

        face: str = raw_tile_face_name.replace("face ", "")
        tile_position_repr, tile_shape_repr = raw_tile_name.replace(
            "lod tile (", "").replace(") with_shape (", "|")[:-1].split("|")

        lod_position: ImgLODPosition = ImgLODPosition.from_string_rep(
            tile_position_repr)

        return CubemapLodPosition(
            pos_pairs=lod_position.pos_pairs,
            face=face
        )
