from dataclasses import dataclass

from boulder_statistics.lods.img_lod_position import ImgLODPosition


@dataclass(frozen=True)
class CubemapLodPosition(ImgLODPosition):
    face: str
