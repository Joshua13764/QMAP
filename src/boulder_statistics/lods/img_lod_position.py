from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from bidict import bidict
from more_itertools import chunked
from numpy.typing import NDArray

BytePair = tuple[bool, bool]
BytePairs = tuple[BytePair, ...]

BYTEPAIR_STR_REP: bidict[BytePair, str] = bidict({
    (False, False): "A",
    (False, True): "B",
    (True, False): "C",
    (True, True): "D",
})

BYTEPAIR_DEBUG_REP: bidict[BytePair, str] = bidict({
    (False, False): "Top Left",
    (False, True): "Top Right",
    (True, False): "Bottom Left",
    (True, True): "Bottom Right",
})

SUBTILE_TOP_LEFT: Dict[str, NDArray[np.float64]] = {
    "A": np.array([0, 0], dtype=np.float64),
    "B": np.array([0.5, 0], dtype=np.float64),
    "C": np.array([0, 0.5], dtype=np.float64),
    "D": np.array([0.5, 0.5], dtype=np.float64),
}


@dataclass(frozen=True)
class ImgLODPosition():
    pos_pairs: BytePairs

    @property
    def string_rep(self) -> str:
        return "".join(BYTEPAIR_STR_REP[pair] for pair in self.pos_pairs)

    @property
    def debug_rep(self) -> str:
        return ", ".join(BYTEPAIR_DEBUG_REP[pair] for pair in self.pos_pairs)

    @property
    def reciprocal_length(self) -> int:
        """
        Returns:
            int: The length of the corner of the cube represented by the pos_pair
        """

        return 2 ** len(self.pos_pairs)

    @property
    def reciprocal_area(self) -> int:
        """
        Returns:
            int: The area of the cube represented by the pos_pair
        """
        return self.reciprocal_length ** 2

    @property
    def lod_number(self) -> int:
        return len(self.pos_pairs)

    @property
    def tile_top_left(self) -> NDArray[np.float64]:
        center: NDArray[np.float64] = np.array([0, 0], dtype=np.float64)

        for i, pair in enumerate(self.pos_pairs):
            str_rep: str = BYTEPAIR_STR_REP[pair]
            subtile_offset: NDArray[np.float64] = SUBTILE_TOP_LEFT[str_rep]
            offset_scale: np.float64 = np.power(2, -i, dtype=np.float64)
            center += subtile_offset * offset_scale

        return center

    @property
    def x_range(self) -> Tuple[float, float]:
        return (self.tile_top_left[0],
                self.tile_top_left[0] + 1 / self.reciprocal_length)

    @property
    def y_range(self) -> Tuple[float, float]:
        return (self.tile_top_left[1],
                self.tile_top_left[1] + 1 / self.reciprocal_length)

    @classmethod
    def parent(cls) -> "ImgLODPosition":
        return cls(pos_pairs=cls.pos_pairs[:-1])

    @classmethod
    def children(cls) -> set["ImgLODPosition"]:
        children_suffixes: Tuple[BytePair, ...] = (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        )

        return set(
            cls(pos_pairs=(*cls.pos_pairs, suffix))
            for suffix in children_suffixes
        )

    @classmethod
    def from_string_rep(cls, str_rep: str) -> "ImgLODPosition":
        pairs: BytePairs = tuple(
            BYTEPAIR_STR_REP.inverse[ch] for ch in str_rep)

        return cls(pos_pairs=pairs)


if __name__ == "__main__":
    test: ImgLODPosition = ImgLODPosition.from_string_rep("CC")
    print(test.tile_top_left)
    print(test.x_range)
    print(test.y_range)
