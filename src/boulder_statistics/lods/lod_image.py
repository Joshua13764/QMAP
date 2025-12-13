from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Callable, Dict, Generic, List, TypeVar

import cv2
import numpy as np

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

T = TypeVar("T", bound=np.floating)


@dataclass
class LODImage(Generic[T]):
    """
    Creates an image which obeys a hierarchical data structure
    """

    lod_tiles: List[LODImageTile[T]]

    def append(self, tile: LODImageTile[T]) -> None:
        self.lod_tiles.append(tile)

    def render_possible_tiles(self, resolution: int) -> None:
        """Gathers the tiles which can be rendered both from one of the provided lod_tiles as well as
        using a combination of all 4 of the sub tiles and add them to its'self
        """

        tiles_to_render: List[ImgLODPosition] = list(LODImage.collect_possible_render_tile_positions(
            set(lod_tile.tile for lod_tile in self.lod_tiles)
        ))

        ordered_tiles_to_render: List[ImgLODPosition] = LODImage.find_tile_render_order(
            current_tiles=[lod_tile.tile for lod_tile in self.lod_tiles],
            tiles_to_render=tiles_to_render
        )

        rendered_lod_tiles: Dict[ImgLODPosition, LODImageTile[T]] = {}

        for tile in ordered_tiles_to_render:

            tile_children: List[LODImageTile[T]] = [
                rendered_lod_tiles[child] for child in tile.children()]

            rendered_lod_tiles[tile] = LODImageTile[T](
                tile=tile,
                get_array_action=lambda: LODImageInterpolator.interpolate_image(
                    resolution, tile_children[0].array, tile_children[1].array, tile_children[2].array,
                    tile_children[3].array
                )
            )

        self.lod_tiles += list(rendered_lod_tiles.values())

    @staticmethod
    def collect_possible_render_tile_positions(
            render_tiles_to_search: set[ImgLODPosition], verbose=True) -> set[ImgLODPosition]:

        found_tiles = render_tiles_to_search.copy()
        number_of_found_tiles_last_last_iteration: int = 0
        number_of_found_tiles_last_iteration: int = len(found_tiles)

        print(
            f"Starting possible render tile search with {number_of_found_tiles_last_iteration} tiles")

        while number_of_found_tiles_last_iteration != number_of_found_tiles_last_last_iteration:

            found_tiles: set[ImgLODPosition] = found_tiles | set([
                tile.parent()
                for tile in render_tiles_to_search
                if tile.parent().children().issubset(render_tiles_to_search)
            ])

            number_of_found_tiles_last_last_iteration = number_of_found_tiles_last_iteration
            number_of_found_tiles_last_iteration = len(found_tiles)

            print(
                f"Found an additional {number_of_found_tiles_last_last_iteration - number_of_found_tiles_last_iteration} tiles")

        return found_tiles

    @staticmethod
    def find_tile_render_order(
            current_tiles: List[ImgLODPosition], tiles_to_render: List[ImgLODPosition]) -> List[ImgLODPosition]:

        tile_to_render_requirements: dict[str, set[str]] = {
            tile.string_rep: (
                set()
                if tile in current_tiles
                else {required_child.string_rep for required_child in tile.children()}
            )
            for tile in tiles_to_render
        }

        ts: TopologicalSorter[str] = TopologicalSorter(
            tile_to_render_requirements)

        tiles_to_render_ordered: List[ImgLODPosition] = [
            ImgLODPosition.from_string_rep(string_rep) for string_rep in ts.static_order()]

        return tiles_to_render_ordered
