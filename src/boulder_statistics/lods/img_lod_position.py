import re
from ast import List
from dataclasses import dataclass
from typing import Tuple

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
