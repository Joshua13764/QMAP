from typing import List, Tuple

from polars import LazyFrame

FACES: List[str] = ["posx", "negx", "posy", "negy", "posz", "negz"]

Pair = Tuple[int, int]
PairGroups = Tuple[Pair, ...]
LazyFileData = tuple[LazyFrame, LazyFrame]
