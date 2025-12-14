from graphlib import TopologicalSorter
from itertools import product
from typing import List

from boulder_statistics.lods.img_lod_position import ImgLODPosition


class LODImageUtils:

    @staticmethod
    def get_all_lod_tiles(depth: int) -> set[ImgLODPosition]:
        iter_chars: List[str] = ["A", "B", "C", "D"]
        combinations: map[str] = map(
            ''.join, product(
                iter_chars, repeat=depth))

        lod_tiles: set[ImgLODPosition] = {ImgLODPosition.from_string_rep(
            combination) for combination in combinations}

        return lod_tiles

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
