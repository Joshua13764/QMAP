from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Callable, Dict, Generic, List, TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.lods.img_lod_position import ImgLODPosition
from boulder_statistics.lods.img_lod_tile import LODImageTile
from boulder_statistics.lods.lod_image_interpolator import LODImageInterpolator
from boulder_statistics.lods.lod_image_utils import LODImageUtils

T = TypeVar("T", bound=np.floating)


@dataclass
class LODImage(Generic[T]):
    """
    Creates an image which obeys a hierarchical data structure
    """
    array_storage_folder_location: FSPathLocalDisk
    array_storage_adapter: FSAdapterBase[NDArray[T], FSPathLocalDisk]

    lod_tiles: List[LODImageTile[T]]

    def append(self, tile: LODImageTile[T]) -> None:
        self.lod_tiles.append(tile)

    def render_possible_tiles(self, resolution: int) -> None:
        """Gathers the tiles which can be rendered both from one of the provided lod_tiles as well as
        using a combination of all 4 of the sub tiles and add them to its'self
        """

        tiles_to_render: List[ImgLODPosition] = list(LODImageUtils.collect_possible_render_tile_positions(
            set(lod_tile.tile for lod_tile in self.lod_tiles)
        ))

        ordered_tiles_to_render: List[ImgLODPosition] = LODImageUtils.find_tile_render_order(
            current_tiles=[lod_tile.tile for lod_tile in self.lod_tiles],
            tiles_to_render=tiles_to_render
        )

        rendered_lod_tiles: Dict[ImgLODPosition, LODImageTile[T]] = {}

        for tile in ordered_tiles_to_render:

            tile_children: List[LODImageTile[T]] = [
                rendered_lod_tiles[child] for child in tile.children()]

            rendered_lod_tiles[tile] = LODImageTile[T](
                tile=tile,
                array_storage_folder_location=self.array_storage_folder_location,
                array_storage_adapter=self.array_storage_adapter,
                array_memory=LODImageInterpolator.interpolate_image(
                    resolution, tile_children[0].array, tile_children[1].array, tile_children[2].array,
                    tile_children[3].array
                )
            )

            # As children only have one parent then we will not be using there
            # arrays for processing immediately therefore they can be unloaded
            # to free up additional memory
            for child in tile_children:
                child.unload_array_from_memory()

        self.lod_tiles += list(rendered_lod_tiles.values())
