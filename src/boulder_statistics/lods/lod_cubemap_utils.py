from typing import Set

from boulder_statistics.lods.cubemap_lod_position import CubemapLodPosition
from boulder_statistics.lods.img_lod_position import ImgLODPosition
from boulder_statistics.lods.lod_image_utils import LODImageUtils
from boulder_statistics.steps.utils.cubemaps_shared import FACES


class LODCubemapUtils:

    @staticmethod
    def get_all_cubemap_tiles_for_depths(
            max_depth: int) -> set[CubemapLodPosition]:
        lod_tiles: Set[ImgLODPosition] = LODImageUtils.get_all_lod_tiles_for_depths(
            max_depth)
        return {
            CubemapLodPosition(
                pos_pairs=tile.pos_pairs,
                face=face,
            )
            for tile in lod_tiles
            for face in FACES
        }
